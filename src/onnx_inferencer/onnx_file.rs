//! Reading one value out of the `.onnx` file itself, without a protobuf library.
//!
//! # Why the runtime reads the graph at all
//!
//! `python -m hf2mobile.postprocess` writes the end-of-sequence ids into the graph as a
//! floating `Constant` node named `hf2mobile_EOS_tokens`. That is the one thing a decode loop
//! needs which is neither an input nor an output: where to stop. Putting it in the graph means
//! a runtime — this one, or an Android app — reads it from the file it already has to open,
//! instead of being handed a json to parse or an argument nobody remembers to pass.
//!
//! ONNX Runtime cannot hand it back: the node has no consumers, so graph resolution prunes it
//! long before a session exists. Hence reading the file directly.
//!
//! # How protobuf makes this easy
//!
//! An `.onnx` file is a protobuf message, which on disk is just a flat sequence of fields.
//! Each field starts with a *tag* holding two numbers: **which** field it is, and **how** its
//! value is encoded (a variable-length integer, a fixed 4 or 8 bytes, or a length followed by
//! that many bytes). Names appear nowhere — `.proto` schemas are a compile-time convenience.
//!
//! The consequence is that a reader can skip any field it does not care about knowing only its
//! wire type, and nested messages are themselves just length-delimited fields. So this file
//! needs no schema and no code generation: the field numbers below, and the ability to seek.
//! That last part is what matters — an export is gigabytes, and this walk reads a few hundred
//! bytes of it.

use std::fs::File;
use std::io::{BufReader, Read};

use anyhow::{bail, Context, Result};

// Field numbers from `onnx.proto3`. A protobuf field is identified by number rather than by
// name, so these are the entire schema as far as this file is concerned. They are part of the
// ONNX format itself and so do not change.
const MODEL_GRAPH: u64 = 7; // ModelProto.graph
const GRAPH_NODE: u64 = 1; // GraphProto.node
const NODE_NAME: u64 = 3; // NodeProto.name
const NODE_OP_TYPE: u64 = 4; // NodeProto.op_type
const NODE_ATTRIBUTE: u64 = 5; // NodeProto.attribute
const ATTR_NAME: u64 = 1; // AttributeProto.name
const ATTR_TENSOR: u64 = 5; // AttributeProto.t
const TENSOR_DATA_TYPE: u64 = 2; // TensorProto.data_type
const TENSOR_INT32_DATA: u64 = 5; // TensorProto.int32_data
const TENSOR_RAW_DATA: u64 = 9; // TensorProto.raw_data
const TENSOR_DATA_LOCATION: u64 = 14; // TensorProto.data_location

// The two `TensorProto.DataType`s an id list can plausibly arrive in.
const INT32: u64 = 6;
const INT64: u64 = 7;

// Wire types: how the bytes after a tag are laid out. (3 and 4 are the deprecated group
// encodings, which protobuf 3 never emits, and 6 and 7 do not exist.)
const VARINT: u8 = 0; // a variable-length integer
const SIXTY_FOUR_BIT: u8 = 1; // exactly 8 bytes
const LENGTH_DELIMITED: u8 = 2; // a length, then that many bytes: strings, bytes, sub-messages
const THIRTY_TWO_BIT: u8 = 5; // exactly 4 bytes

/// The op type the ids are expected to be stored in.
const CONSTANT: &str = "Constant";
/// The attribute a `Constant` node carries its tensor in.
const VALUE: &str = "value";

/// Refuse to allocate for anything larger than this. An id list is a handful of integers, and
/// the length being read here came out of a file, so it needs a ceiling.
const MAX_TENSOR_BYTES: u64 = 64 * 1024;

/// A stretch of the file: byte offsets `[start, end)`.
///
/// Regions are how this file talks about sub-messages without reading them. `Copy` is derived
/// because two integers are cheaper to copy than to reference.
#[derive(Clone, Copy)]
struct Region {
    start: u64,
    end: u64,
}

/// The integers held by the `Constant` node named `node_name`, or `None` if the graph has no
/// node by that name.
///
/// `Some(vec![])` is a real answer, and different from `None`: it is what a `Constant` carrying
/// an empty tensor says, i.e. "this export names no such ids".
pub fn int_constant(path: &str, node_name: &str) -> Result<Option<Vec<i64>>> {
    let file = File::open(path).with_context(|| format!("reading `{path}` to look for `{node_name}`"))?;
    let end = file.metadata().context("measuring the model file")?.len();
    // `BufReader` batches the small reads below into few actual syscalls.
    let mut onnx = Onnx {
        file: BufReader::new(file),
        pos: 0,
        end,
    };

    let whole_file = Region { start: 0, end };
    let graph = onnx
        .submessage(whole_file, MODEL_GRAPH)?
        .with_context(|| format!("`{path}` holds no ONNX graph"))?;
    onnx.constant(graph, node_name)
        .with_context(|| format!("reading `{node_name}` out of `{path}`"))
}

/// A walk over the file that remembers where it is.
///
/// The position is tracked here rather than asked of the OS, so skipping a field is arithmetic
/// plus at most one seek, and reading one is a plain `read_exact`.
struct Onnx {
    file: BufReader<File>,
    /// Offset of the next byte to be read.
    pos: u64,
    /// Length of the file. Every region is checked against it, so a corrupt length cannot send
    /// the walk past the end.
    end: u64,
}

impl Onnx {
    // ── walking the ONNX message ──────────────────────────────────────────────

    /// The body of the first `field` inside `region`, or `None` if it holds no such field.
    ///
    /// Length-delimited fields only, since that is what a nested message is.
    fn submessage(&mut self, region: Region, field: u64) -> Result<Option<Region>> {
        self.goto(region.start)?;
        while self.pos < region.end {
            let (number, wire) = self.tag()?;
            if number == field && wire == LENGTH_DELIMITED {
                return Ok(Some(self.region()?));
            }
            self.skip(wire)?;
        }
        Ok(None)
    }

    /// Scan `graph`'s nodes for the `Constant` named `node_name` and return its integers.
    fn constant(&mut self, graph: Region, node_name: &str) -> Result<Option<Vec<i64>>> {
        self.goto(graph.start)?;
        while self.pos < graph.end {
            let (number, wire) = self.tag()?;
            if number != GRAPH_NODE || wire != LENGTH_DELIMITED {
                self.skip(wire)?;
                continue;
            }
            let node = self.region()?;
            if let Some(values) = self.node_constant(node, node_name)? {
                return Ok(Some(values));
            }
            // `node_constant` stops as soon as it knows this is not the node — usually a few
            // bytes in — so the rest of the node is skipped rather than read.
            self.goto(node.end)?;
        }
        Ok(None)
    }

    /// The integers in `node`, if `node` is the `Constant` called `node_name`.
    ///
    /// `None` means "some other node" and is not a problem. Once the name matches, though,
    /// anything unexpected about the node *is* an error: the graph says it holds these ids, so
    /// failing to read them is not something to paper over with a default.
    fn node_constant(&mut self, node: Region, node_name: &str) -> Result<Option<Vec<i64>>> {
        self.goto(node.start)?;
        // `mut` because both are filled in by the loop below. `None` and `false` are the "we
        // have not seen that field yet" states — protobuf makes no promise that a field is
        // present at all, so every one of them has to have an answer for "absent".
        let mut op_type = None;
        let mut named = false;
        let mut attributes: Vec<Region> = Vec::new();

        while self.pos < node.end {
            let (number, wire) = self.tag()?;
            match (number, wire) {
                (NODE_NAME, LENGTH_DELIMITED) => {
                    // Fields are written in field-number order, so the name (3) arrives before
                    // the attributes (5). That ordering is what keeps this walk off the
                    // weights of every other `Constant` in the graph.
                    if self.text()? != node_name {
                        return Ok(None);
                    }
                    named = true;
                }
                (NODE_OP_TYPE, LENGTH_DELIMITED) => op_type = Some(self.text()?),
                (NODE_ATTRIBUTE, LENGTH_DELIMITED) => {
                    // Noted down as offsets and revisited below, rather than read now: at this
                    // point we may not know yet whether this is even the right node.
                    let attribute = self.region()?;
                    attributes.push(attribute);
                    self.goto(attribute.end)?;
                }
                // Every other field — inputs, outputs, the domain — is stepped over using its
                // wire type alone, which is the whole reason this file needs no schema.
                _ => self.skip(wire)?,
            }
        }

        // `NodeProto.name` is optional, and protobuf 3 does not write a field holding the
        // default value — so a node named `""` arrives here with no name field at all and the
        // match above never runs. Without this, such a node would fall into the `op_type`
        // check below and fail the whole read with "expected a `Constant` node, found `Add`".
        if !named {
            return Ok(None);
        }

        if op_type.as_deref() != Some(CONSTANT) {
            bail!(
                "expected a `{CONSTANT}` node, found `{}`",
                op_type.as_deref().unwrap_or("<unnamed op>")
            );
        }
        for attribute in attributes {
            if let Some(tensor) = self.tensor_attribute(attribute, VALUE)? {
                return self.int_tensor(tensor).map(Some);
            }
        }
        bail!("the node carries no `{VALUE}` attribute holding a tensor")
    }

    /// The tensor in `attribute`, if `attribute` is the one called `name`.
    fn tensor_attribute(&mut self, attribute: Region, name: &str) -> Result<Option<Region>> {
        self.goto(attribute.start)?;
        let mut matched = false;
        let mut tensor = None;

        while self.pos < attribute.end {
            let (number, wire) = self.tag()?;
            match (number, wire) {
                (ATTR_NAME, LENGTH_DELIMITED) => matched = self.text()? == name,
                (ATTR_TENSOR, LENGTH_DELIMITED) => {
                    let region = self.region()?;
                    tensor = Some(region);
                    self.goto(region.end)?;
                }
                _ => self.skip(wire)?,
            }
        }
        // `filter` throws the region away unless the name matched, whichever order they came in.
        Ok(tensor.filter(|_| matched))
    }

    /// The integers held by a `TensorProto`, widened to `i64`.
    ///
    /// int32 and int64 only: those are what an id list is written as, and a float tensor here
    /// would mean the graph is saying something other than what we came to read.
    fn int_tensor(&mut self, tensor: Region) -> Result<Vec<i64>> {
        self.goto(tensor.start)?;
        let mut data_type = None;
        let mut raw = None;
        let mut packed = None;
        let mut external = false;

        while self.pos < tensor.end {
            let (number, wire) = self.tag()?;
            match (number, wire) {
                (TENSOR_DATA_TYPE, VARINT) => data_type = Some(self.varint()?),
                (TENSOR_DATA_LOCATION, VARINT) => external = self.varint()? != 0,
                (TENSOR_RAW_DATA, LENGTH_DELIMITED) => {
                    let region = self.region()?;
                    raw = Some(region);
                    self.goto(region.end)?;
                }
                (TENSOR_INT32_DATA, LENGTH_DELIMITED) => {
                    let region = self.region()?;
                    packed = Some(region);
                    self.goto(region.end)?;
                }
                _ => self.skip(wire)?,
            }
        }

        if external {
            bail!("its tensor lives in external data, which this reader does not follow");
        }
        let width = match data_type {
            Some(INT32) => 4,
            Some(INT64) => 8,
            other => bail!(
                "its tensor has ONNX element type {}, and only int32 ({INT32}) and int64 ({INT64}) hold ids",
                other.map_or_else(|| "<unset>".to_string(), |ty| ty.to_string())
            ),
        };

        // `raw_data` is the case that happens, because that is what
        // `onnx.numpy_helper.from_array` writes. `int32_data` is handled too, because a graph
        // built by hand with `onnx.helper.make_tensor` legitimately uses it instead.
        if let Some(region) = raw {
            let bytes = self.bytes(region)?;
            if bytes.len() % width != 0 {
                bail!(
                    "its raw_data is {} bytes, not a whole number of {width}-byte ids",
                    bytes.len()
                );
            }
            // `chunks_exact` hands out non-overlapping windows of exactly `width` bytes.
            return Ok(bytes.chunks_exact(width).map(little_endian).collect());
        }
        if let Some(region) = packed {
            self.goto(region.start)?;
            let mut values = Vec::new();
            while self.pos < region.end {
                // A negative int32 is varint-encoded as its 64-bit sign extension, so reading
                // it as a 64-bit value and reinterpreting is correct at either width.
                values.push(self.varint()? as i64);
            }
            return Ok(values);
        }
        // A tensor with no data at all: an empty id list, which postprocess writes when the
        // model names no stop token.
        Ok(Vec::new())
    }

    // ── the protobuf wire format ──────────────────────────────────────────────

    /// Field number and wire type of the next field, from one varint: the low three bits are
    /// the wire type, everything above them is the field number.
    fn tag(&mut self) -> Result<(u64, u8)> {
        let key = self.varint()?;
        Ok((key >> 3, (key & 0b111) as u8))
    }

    /// A base-128 varint: seven bits of the number per byte, lowest bits first, and the top bit
    /// of each byte set on every byte except the last.
    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        bail!("a varint runs past ten bytes, so this is not an ONNX file")
    }

    /// The region a length-delimited field's body occupies, leaving the position at its start.
    /// Bounded by the file length, so a nonsense length is caught here rather than turning into
    /// a wild read.
    fn region(&mut self) -> Result<Region> {
        let len = self.varint()?;
        let start = self.pos;
        // `checked_add` returns `None` instead of wrapping around on overflow, and `filter`
        // then also rejects any end past the file — either way `with_context` turns the `None`
        // into the error below.
        let end = start
            .checked_add(len)
            .filter(|&end| end <= self.end)
            .with_context(|| format!("a field at offset {start} claims {len} bytes, past the end of the file"))?;
        Ok(Region { start, end })
    }

    /// The current length-delimited field, as a `String`.
    fn text(&mut self) -> Result<String> {
        let region = self.region()?;
        let bytes = self.bytes(region)?;
        String::from_utf8(bytes).context("a name in the graph is not valid UTF-8")
    }

    /// Read `region` into memory. The only place this file allocates from a length it read.
    fn bytes(&mut self, region: Region) -> Result<Vec<u8>> {
        let len = region.end - region.start;
        if len > MAX_TENSOR_BYTES {
            bail!("a field of {len} bytes is far larger than the id list this reader expects");
        }
        self.goto(region.start)?;
        let mut bytes = vec![0u8; len as usize];
        self.file.read_exact(&mut bytes).context("unexpected end of file")?;
        self.pos = region.end;
        Ok(bytes)
    }

    fn byte(&mut self) -> Result<u8> {
        let mut byte = [0u8];
        self.file.read_exact(&mut byte).context("unexpected end of file")?;
        self.pos += 1;
        Ok(byte[0])
    }

    /// Step over a field whose contents we do not need, given its wire type.
    fn skip(&mut self, wire: u8) -> Result<()> {
        match wire {
            VARINT => {
                self.varint()?;
            }
            SIXTY_FOUR_BIT => self.goto(self.pos + 8)?,
            THIRTY_TWO_BIT => self.goto(self.pos + 4)?,
            LENGTH_DELIMITED => {
                let region = self.region()?;
                self.goto(region.end)?;
            }
            other => bail!("wire type {other} does not appear in an ONNX file"),
        }
        Ok(())
    }

    /// Move to an absolute offset.
    fn goto(&mut self, pos: u64) -> Result<()> {
        if pos == self.pos {
            return Ok(());
        }
        // Relative rather than absolute seeking, because `BufReader::seek_relative` keeps the
        // bytes it has already buffered when the target is still among them — which is the
        // common case here, as the fields this walk skips are short.
        self.file
            .seek_relative(pos as i64 - self.pos as i64)
            .with_context(|| format!("seeking to offset {pos}"))?;
        self.pos = pos;
        Ok(())
    }
}

/// One little-endian integer, at whichever of the two widths `chunk` is.
///
/// `try_into` converts the slice to a fixed-size array, which is what `from_le_bytes` needs.
/// It cannot fail here — `chunks_exact` guarantees the length — and `expect` is how that
/// reasoning is written down.
fn little_endian(chunk: &[u8]) -> i64 {
    match chunk.len() {
        4 => i64::from(i32::from_le_bytes(chunk.try_into().expect("four bytes"))),
        _ => i64::from_le_bytes(chunk.try_into().expect("eight bytes")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The encoder side of the format, so the tests can build a graph and read it back. Only
    // the two wire types an ONNX message needs.

    fn varint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 0x80);
        }
    }

    fn field(number: u64, wire: u8, body: &[u8]) -> Vec<u8> {
        let mut out = varint(number << 3 | u64::from(wire));
        if wire == LENGTH_DELIMITED {
            out.extend(varint(body.len() as u64));
        }
        out.extend_from_slice(body);
        out
    }

    fn text(number: u64, value: &str) -> Vec<u8> {
        field(number, LENGTH_DELIMITED, value.as_bytes())
    }

    /// A `ModelProto` holding one `Constant` node per `(name, ids)` pair.
    ///
    /// An empty name is the exception: it stands for an *unnamed* node, which on disk means
    /// no `NodeProto.name` field at all (protobuf 3 does not write a default value), and is
    /// what `onnx.helper.make_node(...)` produces when the caller passes no `name=`.
    fn model(nodes: &[(&str, &[i32])]) -> Vec<u8> {
        let mut graph = Vec::new();
        for (name, ids) in nodes {
            if name.is_empty() {
                let mut node = text(2, "unnamed_output"); // NodeProto.output
                node.extend(text(NODE_OP_TYPE, "Add"));
                graph.extend(field(GRAPH_NODE, LENGTH_DELIMITED, &node));
                continue;
            }
            let raw: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            let mut tensor = field(1, VARINT, &varint(ids.len() as u64)); // dims
            tensor.extend(field(TENSOR_DATA_TYPE, VARINT, &varint(INT32)));
            tensor.extend(text(8, name)); // TensorProto.name
            tensor.extend(field(TENSOR_RAW_DATA, LENGTH_DELIMITED, &raw));

            let mut attribute = text(ATTR_NAME, VALUE);
            attribute.extend(field(ATTR_TENSOR, LENGTH_DELIMITED, &tensor));

            let mut node = text(2, name); // NodeProto.output
            node.extend(text(NODE_NAME, name));
            node.extend(text(NODE_OP_TYPE, CONSTANT));
            node.extend(field(NODE_ATTRIBUTE, LENGTH_DELIMITED, &attribute));
            graph.extend(field(GRAPH_NODE, LENGTH_DELIMITED, &node));
        }

        let mut model = field(1, VARINT, &varint(10)); // ir_version
        model.extend(text(2, "hf2mobile")); // producer_name
        model.extend(field(MODEL_GRAPH, LENGTH_DELIMITED, &graph));
        model
    }

    fn write(bytes: &[u8]) -> std::path::PathBuf {
        // A path per test process and per call, so a parallel run cannot collide.
        static COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let nth = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("hf2mobile_onnx_{}_{nth}.onnx", std::process::id()));
        std::fs::write(&path, bytes).expect("writing the fixture");
        path
    }

    fn read(bytes: &[u8], node_name: &str) -> Result<Option<Vec<i64>>> {
        int_constant(write(bytes).to_str().expect("utf-8 temp path"), node_name)
    }

    #[test]
    fn reads_the_ids_out_of_a_named_constant() {
        let file = model(&[("hf2mobile_EOS_tokens", &[1, 106])]);
        assert_eq!(read(&file, "hf2mobile_EOS_tokens").unwrap(), Some(vec![1, 106]));
    }

    #[test]
    fn finds_a_constant_that_is_not_the_first_node() {
        // The decoy's tensor is deliberately bigger than `MAX_TENSOR_BYTES`, so a walk that
        // read it instead of skipping it would fail rather than quietly cost time.
        let file = model(&[("other", &[7; 32768]), ("hf2mobile_EOS_tokens", &[151645])]);
        assert_eq!(
            read(&file, "hf2mobile_EOS_tokens").unwrap(),
            Some(vec![151645]),
            "a large earlier node must be skipped, not read"
        );
    }

    #[test]
    fn an_empty_tensor_is_an_empty_list_not_a_missing_node() {
        let file = model(&[("hf2mobile_EOS_tokens", &[])]);
        assert_eq!(
            read(&file, "hf2mobile_EOS_tokens").unwrap(),
            Some(vec![]),
            "the export names no stop token, which is not the same as no node"
        );
    }

    #[test]
    fn a_missing_node_is_none() {
        let file = model(&[("something_else", &[1])]);
        assert_eq!(read(&file, "hf2mobile_EOS_tokens").unwrap(), None);
    }

    /// A node carrying no name at all is another node, not a broken `Constant`. Reading it as
    /// one used to fail the whole walk, which would have taken the model down with it.
    #[test]
    fn an_unnamed_node_is_skipped_rather_than_failing_the_walk() {
        let file = model(&[("", &[]), ("hf2mobile_EOS_tokens", &[151645])]);
        assert_eq!(read(&file, "hf2mobile_EOS_tokens").unwrap(), Some(vec![151645]));
        assert_eq!(read(&model(&[("", &[])]), "hf2mobile_EOS_tokens").unwrap(), None);
    }

    #[test]
    fn truncated_and_absent_files_fail_rather_than_guess() {
        let file = model(&[("hf2mobile_EOS_tokens", &[1, 106])]);
        assert!(read(&file[..file.len() - 3], "hf2mobile_EOS_tokens").is_err());
        assert!(int_constant("/nonexistent/model.onnx", "hf2mobile_EOS_tokens").is_err());
    }
}

{
  description = "HF2Mobile";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = { self, nixpkgs, rust-overlay }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
      overlays = [ rust-overlay.overlays.default ];
    };

    # Upstream rust dist tarballs instead of nixpkgs' rustc/cargo: the latter drag in rustc-bootstrap + llvm-lib (~590MB extra download).
    rustToolchain = pkgs.rust-bin.stable.latest.minimal.override {
      extensions = [ "rustfmt" "rust-analyzer" "rust-src" ];
      # The phone's std, so `cargo build --target aarch64-linux-android` works
      targets = [ "aarch64-linux-android" ];
    };

    # The NDK supplies the C compiler that links an Android binary (and builds the oniguruma inside `tokenizers`);
    # platform-tools supplies `adb` to push the result.
    # Both live in the `android` shell only — they are a multi-GB download that the day-to-day shell has no use for.
    # `.cargo/config.toml` names the compilers this puts on $PATH.
    androidNdk = pkgs.androidenv.androidPkgs.ndk-bundle;
    androidToolchainBin = "${androidNdk}/libexec/android-sdk/ndk-bundle/toolchains/llvm/prebuilt/linux-x86_64/bin";

    envVars = {
      UV_PYTHON_DOWNLOADS = "never";
    };
  in {
    devShells.${system} = {
    # `nix develop`           — Python + Rust, everything the host workflow needs.
    # `nix develop .#android` — the cross-compilation shell defined below.
    default =
      pkgs.mkShell {
        nativeBuildInputs = [];
        buildInputs = with pkgs; [
          # Python
          python312
          uv
          pyright

          # Rust
          rustToolchain
          maturin
        ];
        env = envVars;
        packages = [];
        shellHook = ''
        export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib pkgs.zlib]}:$LD_LIBRARY_PATH"

        export PRJ_ROOT="$PWD"
        export HF_HOME="$PRJ_ROOT/.hf_cache"
        export UV_CACHE_DIR="$PRJ_ROOT/.uv_cache"
        export CARGO_HOME="$PRJ_ROOT/.cargo"

        if [ ! -d ".venv" ]; then
          echo "Creating virtual environment..."
          uv venv --python ${pkgs.python312}/bin/python
        fi
        source .venv/bin/activate
        echo "Python Venv Activated!"
        '';
      };

    # Cross-compiling the `hf2mobile-infer` CLI for a phone. No Python: this shell builds a
    # binary and pushes it, and the export it runs was made in the shell above.
    #
    #   nix develop .#android -c cargo build --release --target aarch64-linux-android --bin hf2mobile-infer
    android =
      pkgs.mkShell {
        buildInputs = [
          rustToolchain
          androidNdk
          pkgs.androidenv.androidPkgs.platform-tools  # adb
        ];
        shellHook = ''
        export PATH="${androidToolchainBin}:$PATH"
        export ANDROID_NDK_HOME="${androidNdk}/libexec/android-sdk/ndk-bundle"
        export CARGO_HOME="$PWD/.cargo"
        echo "Android NDK $(basename ${androidNdk}) — cross toolchain and adb on PATH"
        '';
      };
    };
  };
}

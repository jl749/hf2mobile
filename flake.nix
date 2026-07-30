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
    };

    envVars = {
      UV_PYTHON_DOWNLOADS = "never";
    };
  in {
    devShells.${system}.default =
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
  };
}

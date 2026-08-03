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

    envVars = {
      UV_PYTHON_DOWNLOADS = "never";
    };

    # The NDK supplies the C compiler that links an Android binary (and builds the oniguruma inside `tokenizers`);
    # platform-tools supplies `adb` to push the result.
    # Both live in the `android` shell only — they are a multi-GB download that the day-to-day shell has no use for.
    # `.cargo/config.toml` names the compilers this puts on $PATH.
    androidNdk = pkgs.androidenv.androidPkgs.ndk-bundle;
    androidToolchainBin = "${androidNdk}/libexec/android-sdk/ndk-bundle/toolchains/llvm/prebuilt/linux-x86_64/bin";

    commonShellHook = ''
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
  in {
    devShells.${system} = {
    # `nix develop`           — Python + Rust base environment
    # `nix develop .#android` — extend default environment for Android deployment
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
        shellHook = commonShellHook;
      };

    android =
      pkgs.mkShell {
        buildInputs = with pkgs; [
          # Python
          python312
          uv
          pyright

          # Rust
          rustToolchain
          maturin

          # Android
          androidNdk
          androidenv.androidPkgs.platform-tools  # adb
        ];
        env = envVars;
        shellHook = commonShellHook + ''

          export PATH="${androidToolchainBin}:$PATH"
          export ANDROID_NDK_HOME="${androidNdk}/libexec/android-sdk/ndk-bundle"
          echo "Android NDK $(basename ${androidNdk}) — cross toolchain and adb on PATH"
        '';
      };
    };
  };
}

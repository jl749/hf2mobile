{
  description = "PYTHON ENV";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
    envVars = {
      HF_HOME = "./.hf_cache";
      UV_CACHE_DIR = "./.uv_cache";
      UV_PYTHON_DOWNLOADS = "never";
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
    };
  in {
    devShells.${system}.default = 
      pkgs.mkShell {
        nativeBuildInputs = [];
        buildInputs = with pkgs; [
          pkgs.python312
          pkgs.pyright
          pkgs.uv
        ];
        env = envVars;
        packages = [];
        shellHook = ''
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

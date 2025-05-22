{
  description = "Projeto que estende o ambiente Essentials";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    essentials.url = "git+file:///mnt/hdmenezess42/GitProjects/flakeEssentials";
  };

  outputs = { self, nixpkgs, flake-utils, essentials }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        baseShell = essentials.devShells.${system}.python;
      in {
        devShell = pkgs.mkShell {
          name = "projeto-com-requests";

          buildInputs = baseShell.buildInputs ++ (with pkgs.python311Packages; [
          # opencv4
          pandas
          scikit-learn
          seaborn
          rich
          prompt-toolkit
          tensorflow
          keras
          ]);

          shellHook = ''
            echo "Ambiente do projeto carregado (base Essentials + customizações)."
            ${baseShell.shellHook or ""}
          '';
        };
      });
}

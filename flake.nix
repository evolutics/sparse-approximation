{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
    travel-kit.url = "github:evolutics/travel-kit";
  };

  outputs = inputs @ {
    flake-utils,
    nixpkgs,
    travel-kit,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};
    in {
      devShell = pkgs.mkShellNoCC {
        buildInputs =
          (with pkgs; [
            python311Full
            python311Packages.altair
            python311Packages.cvxpy
            python311Packages.ipywidgets
            python311Packages.jupyterlab
            python311Packages.jupytext
            python311Packages.numpy
            python311Packages.pandas
            python311Packages.pytest
            python311Packages.scipy
          ])
          ++ [travel-kit.defaultApp.${system}];

        shellHook = ''
          export PYTHONPATH="$PWD:$PYTHONPATH"
        '';
      };
    });
}

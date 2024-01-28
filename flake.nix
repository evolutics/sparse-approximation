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
      pythonPackages = ps:
        with ps; [
          altair
          cvxpy
          ipywidgets
          jupyterlab
          jupytext
          numpy
          pandas
          pytest
          scipy
        ];
    in {
      devShell = pkgs.mkShellNoCC {
        buildInputs =
          (with pkgs; [
            (python3.withPackages pythonPackages)
          ])
          ++ [travel-kit.defaultApp.${system}];

        shellHook = ''
          export PYTHONPATH="$PWD:$PYTHONPATH"
        '';
      };
    });
}

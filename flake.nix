{
  # flake.nix template for Rust projects with Fenix
  description = "Rust devShell for <project-name>";

  inputs = {
    # Follow nixpkgs unstable for latest packages
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # Fenix for Rust toolchains
    fenix.url = "github:nix-community/fenix";
    # Optional: rust-overlay for extra targets
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    fenix,
    rust-overlay,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        # Import pkgs with overlays for Fenix and rust-overlay
        pkgs = import nixpkgs {
          inherit system;
          overlays = [fenix.overlays.default rust-overlay.overlays.default];
        };
        inherit (pkgs) lib;

        # Define Rust toolchain and analyzer from Fenix
        stableToolchain = fenix.packages.${system}.complete.toolchain;
        rustAnalyzer = fenix.packages.${system}.latest.rust-analyzer;
      in {
        # Default devShell for project development
        devShells.default = pkgs.mkShell {
          name = "burn-shell";

          buildInputs = with pkgs;
            lib.flatten [
              # Core Rust toolchain
              stableToolchain
              # IDE support
              rustAnalyzer
              # Common Rust utilities
              cargo-expand
              # Optional REPL shell
              nushell
            ];

          # Customize shell behavior here
          shellHook = ''
            echo "Rust version: $(rustc --version)";
            # Keep cargo cache in home
            export CARGO_HOME="$HOME/.cargo";
            export RUSTUP_HOME="$HOME/.rustup";
            mkdir -p "$CARGO_HOME" "$RUSTUP_HOME";
          '';
        };

        # Optionally expose a default package
        defaultPackage = stableToolchain;
      }
    );
}

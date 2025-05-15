{
  description =
    "Lecture Video Content Indexer - Educational video analysis with theory/practice classification";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true; # In case we need non-free dependencies
        };

        pythonEnv = pkgs.python3.withPackages (ps:
          with ps; [
            # misc
            colorama
            pybind11
            pip

            # Core dependencies
            fastapi
            uvicorn
            pydantic
            aiofiles
            httpx
            sqlalchemy

            # Data processing
            nltk
            scikit-learn
            numpy
            pandas
            spacy
            spacy-models.en_core_web_sm
            spacy-models.ru_core_news_sm
            rapidfuzz

            # Google API
            google-api-python-client
            google-auth
            youtube-transcript-api

            # Utilities
            pyyaml
            python-dotenv
            tqdm
            matplotlib

            # Testing
            pytest
            pytest-asyncio
            pytest-cov

            # Type checking and linting
            mypy
            pylint
            black
            isort

            langdetect
            git-filter-repo
            tabulate
          ]);

        # Download spaCy models
        spacyModels = pkgs.runCommand "spacy-models" { } ''
          mkdir -p $out/bin

          # Create a script to download models if needed
          cat > $out/bin/download-spacy-models <<EOF
          #!/bin/sh
          ${pythonEnv}/bin/python -m spacy download en_core_web_sm
          ${pythonEnv}/bin/python -m spacy download ru_core_news_sm
          EOF

          chmod +x $out/bin/download-spacy-models
        '';

        # Download NLTK data
        nltkData = pkgs.runCommand "nltk-data" { } ''
          mkdir -p $out/bin

          # Create a script to download NLTK data if needed
          cat > $out/bin/download-nltk-data <<EOF
          #!/bin/sh
          ${pythonEnv}/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
          EOF

          chmod +x $out/bin/download-nltk-data
        '';

        # Create a script to run the API server
        apiServer = pkgs.writeScriptBin "lecture-indexer-api" ''
          #!/bin/sh
          cd $PWD
          ${pythonEnv}/bin/uvicorn integration.api_service.python.api_service:app --host 0.0.0.0 --port 8080 "$@"
        '';

      in {
        packages = {
          default = self.packages.${system}.lecture-indexer;

          lecture-indexer = pkgs.stdenv.mkDerivation {
            pname = "lecture-indexer";
            version = "1.0.0";
            src = ./.;

            buildInputs = [ pythonEnv ];

            buildPhase = ''
              mkdir -p $out/bin
              mkdir -p $out/lib/python3.10/site-packages/lecture_indexer
              cp -r . $out/lib/python3.10/site-packages/lecture_indexer

              # Create wrapper script
              cat > $out/bin/lecture-indexer <<EOF
              #!/bin/sh
              PYTHONPATH=$out/lib/python3.10/site-packages:$PYTHONPATH ${pythonEnv}/bin/python -m lecture_indexer.main "\$@"
              EOF

              chmod +x $out/bin/lecture-indexer
            '';

            installPhase = ''
              # Nothing additional to do
            '';
          };

          spacyModels = spacyModels;
          nltkData = nltkData;
          apiServer = apiServer;
        };

        apps = {
          default = self.apps.${system}.lecture-indexer-api;

          lecture-indexer-api = {
            type = "app";
            program = "${apiServer}/bin/lecture-indexer-api";
          };

          download-spacy-models = {
            type = "app";
            program = "${spacyModels}/bin/download-spacy-models";
          };

          download-nltk-data = {
            type = "app";
            program = "${nltkData}/bin/download-nltk-data";
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.sqlite

            # Development tools
            pkgs.pre-commit
            pkgs.git
          ];

          shellHook = ''
            export PYTHONPATH="$PWD:$PYTHONPATH"

            # Pass through YouTube API key from host environment if available
            if [ -n "$YOUTUBE_API_KEY" ]; then
              echo "Found YOUTUBE_API_KEY in environment, using it for configuration"
            else
              echo "Warning: YOUTUBE_API_KEY is not set in environment"
              # You can set a default key for development here if needed
              # export YOUTUBE_API_KEY="AIzaSyA5n2hS2aVYrm7HeHP7u0iM7ubOyVTGQ-o"
            fi

            # Create data directories if they don't exist
            mkdir -p data/processed
            mkdir -p data/index
            mkdir -p data/tasks
            mkdir -p data/results

            # Create config directory if it doesn't exist
            mkdir -p config

            # Create default config files if they don't exist
            if [ ! -f config/api.yaml ]; then
              echo "Creating default API config..."
              cat > config/api.yaml <<EOF
            youtube_api_key: "\''${YOUTUBE_API_KEY}"
            task_dir: "data/tasks"
            result_dir: "data/results"
            max_workers: 4
            EOF
            fi

            if [ ! -f config/pipeline.yaml ]; then
              echo "Creating default pipeline config..."
              cat > config/pipeline.yaml <<EOF
            youtube_api_key: "\''${YOUTUBE_API_KEY}"
            output_dir: "data/processed"
            EOF
            fi

            if [ ! -f config/search.yaml ]; then
              echo "Creating default search config..."
              cat > config/search.yaml <<EOF
            index_dir: "data/index"
            EOF
            fi

            # Check if configs exist but have empty API keys, and the env var is available
            if [ -n "$YOUTUBE_API_KEY" ]; then
              # Update api.yaml if it exists but has empty API key
              if [ -f config/api.yaml ] && ! grep -q "youtube_api_key: \"\''${YOUTUBE_API_KEY}\"" config/api.yaml; then
                echo "Updating API key in config/api.yaml..."
                # Use sed to replace the API key line with the environment variable reference
                sed -i "s|youtube_api_key:.*|youtube_api_key: \"\''${YOUTUBE_API_KEY}\"|" config/api.yaml
              fi

              # Update pipeline.yaml if it exists but has empty API key
              if [ -f config/pipeline.yaml ] && ! grep -q "youtube_api_key: \"\''${YOUTUBE_API_KEY}\"" config/pipeline.yaml; then
                echo "Updating API key in config/pipeline.yaml..."
                sed -i "s|youtube_api_key:.*|youtube_api_key: \"\''${YOUTUBE_API_KEY}\"|" config/pipeline.yaml
              fi
            fi

            echo "Development environment ready!"
            echo "Run 'nix run .#download-spacy-models' to download required spaCy models"
            echo "Run 'nix run .#download-nltk-data' to download required NLTK data"
            echo "Run 'nix run .#lecture-indexer-api' to start the API server"

            # Display a helpful message about the YouTube API key
            if [ -z "$YOUTUBE_API_KEY" ]; then
              echo ""
              echo "************************************************************"
              echo "IMPORTANT: Set your YouTube API key before running the application:"
              echo "export YOUTUBE_API_KEY='your-api-key'"
              echo "************************************************************"
            else
              echo ""
              echo "************************************************************"
              echo "Using YOUTUBE_API_KEY from environment"
              echo "************************************************************"
            fi
          '';
        };
      });
}

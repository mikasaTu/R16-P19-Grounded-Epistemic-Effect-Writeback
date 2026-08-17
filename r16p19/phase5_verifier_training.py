"""CLI-compatible verifier training entry point."""

from .phase5_verifier_model import main, train

__all__ = ["train", "main"]

if __name__ == "__main__":
    main()

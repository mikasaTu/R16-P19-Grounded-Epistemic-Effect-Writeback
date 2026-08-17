"""Public entry point for the bounded systems benchmark."""

from .phase5_bounded_benchmark import main, run

__all__ = ["run", "main"]

if __name__ == "__main__":
    main()

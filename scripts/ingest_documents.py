"""
Bulk-ingest every supported file under a directory (default: data/raw).

    python -m scripts.ingest_documents
    python -m scripts.ingest_documents path/to/docs --tenant acme --strategy fixed

Documents are registered as tenant-visible documents of the given tenant.
"""
import argparse
import json
import sys

from app.config import settings
from app.logging_config import configure_logging
from app.rag import EnterpriseRAG


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", default="data/raw")
    parser.add_argument("--tenant", default=settings.DEFAULT_TENANT)
    parser.add_argument("--strategy", choices=["fixed", "recursive"], default=None)
    args = parser.parse_args(argv)

    configure_logging()
    rag = EnterpriseRAG()
    try:
        summary = rag.ingest_directory(args.directory, args.strategy, tenant_id=args.tenant)
    except FileNotFoundError:
        print(f"error: {args.directory} does not exist.", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    print(f"Total indexed chunks: {rag.stats()['total_chunks']}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

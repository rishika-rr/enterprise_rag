"""
Create a user from the command line (operators only; needs access to the server).

    python -m scripts.create_user alice --role document_manager
    python -m scripts.create_user root@acme --role administrator --tenant acme

The password is read from a prompt (or the RAG_USER_PASSWORD environment
variable for automation) so it never appears in shell history.
This is also the only way to create the first administrator of a new tenant.
"""
import argparse
import getpass
import os
import sys

from app.config import settings
from app.database.state import get_state_store
from app.models.identity import Role


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("username")
    parser.add_argument("--role", choices=[r.value for r in Role], default=Role.READER.value)
    parser.add_argument("--tenant", default=settings.DEFAULT_TENANT)
    args = parser.parse_args(argv)

    password = os.getenv("RAG_USER_PASSWORD") or getpass.getpass("Password: ")
    store = get_state_store()
    try:
        user = store.create_user(args.username, password, Role(args.role), args.tenant)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    store.audit("user.create", "success", target=user.id, tenant_id=user.tenant_id,
                actor_name="cli", detail=f"role={user.role.value}")
    print(f"created {user.username} ({user.role.value}, tenant {user.tenant_id}) id={user.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

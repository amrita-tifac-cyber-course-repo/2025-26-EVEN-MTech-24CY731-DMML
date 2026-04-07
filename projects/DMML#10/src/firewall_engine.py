ALLOW = 0
BLOCK = 1
RATE_LIMIT = 2
BLOCK_SRC = 3
BLOCK_PORT = 4

ACTION_NAMES = {
    0: "ALLOW",
    1: "BLOCK",
    2: "RATE_LIMIT",
    3: "BLOCK_SRC",
    4: "BLOCK_PORT"
}

class Firewall:
    def __init__(self):
        self.blocked_ports = set()
        self.blocked_sources = set()

    def apply(self, action, row):
        port = int(row["Destination_Port"])
        src_id = row.name  # simulated source identity

        # Already blocked
        if port in self.blocked_ports or src_id in self.blocked_sources:
            return "BLOCK"

        if action == ALLOW:
            return "ALLOW"

        elif action == BLOCK:
            return "BLOCK"

        elif action == RATE_LIMIT:
            return "RATE_LIMIT"

        elif action == BLOCK_SRC:
            self.blocked_sources.add(src_id)
            return "BLOCK"

        elif action == BLOCK_PORT:
            self.blocked_ports.add(port)
            return "BLOCK"

        return "ALLOW"

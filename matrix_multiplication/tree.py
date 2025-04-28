class TreeNode:
    def __init__(self, size, base_case=False, padded=False):
        self.size = size
        self.padded = padded
        self.base_case = base_case
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def display(self, prefix="", is_last=True, is_root=True):
        connector = ""
        new_prefix = ""
        if not is_root:
            connector = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")
        size_str = f"({str(self.size)}+1)" if self.padded else str(self.size)
        print(prefix + connector + f"{size_str}x{size_str}{' (base case)' if self.base_case else ''}")

        for i, child in enumerate(self.children):
            is_last_child = (i == len(self.children) - 1)
            child.display(prefix=new_prefix, is_last=is_last_child, is_root=False)

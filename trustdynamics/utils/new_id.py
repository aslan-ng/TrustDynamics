import uuid


def new_unique_id(existing_values: list[int] | tuple[int] | set[int] = []) -> int:
    while True:
        new = uuid.uuid4().int
        if new not in existing_values:
            return new
        

if __name__ == "__main__":
    existing_values = set([1, 2])
    new_id = new_unique_id(existing_values)
    print(new_id)
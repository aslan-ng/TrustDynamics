import uuid


def new_unique_id_uuid(existing_values: list[int] | tuple[int] | set[int] = []) -> int:
    while True:
        new = uuid.uuid4().int
        if new not in existing_values:
            return new
        
def new_unique_id_maxval(existing_values: list[int] | tuple[int] | set[int] = []) -> int:
    if not existing_values:
        return 1
    else:
        return max(existing_values) + 1
        

if __name__ == "__main__":
    existing_values = set([1, 2])
    new_id = new_unique_id_maxval(existing_values)
    print(new_id)
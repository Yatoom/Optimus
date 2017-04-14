def get_openml_splits(task):
    generator = task.iterate_all_splits()
    splits = [(i.train, i.test) for i in generator]
    return splits
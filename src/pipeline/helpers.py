def flatten_list(list_to_flatten):
    flat_list = []
    for sublist in list_to_flatten:
        for item in sublist:
            flat_list.append(item)



def remove_suffix(filename, file_to_write):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('%s' % filename, encoding='utf-8').\
        read().strip().split('\n')

    trans = open('%s' % file_to_write, 'w')

    for str in lines:
        length = len(str)
        str = str[0: length - 6]
        trans.write(str + '\n')

    trans.close()


# remove_suffix('origin_data/origin_trans.txt', 'origin_data/trans.txt')
remove_suffix('simulation_data/simulation_trans.txt', 'simulation_data/trans.txt')

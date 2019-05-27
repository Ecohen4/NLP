import pandas as pd

def load_char_list(csv):
    char_data = pd.read_csv(csv)
    char_list = char_data.name.tolist()
    # split all f,l names separately
    char_names_all = []
    for term in char_list:
        all_terms = term.split(' ')
        for x in all_terms:
            char_names_all.append(x)
    # clean up
    rm_terms = ['Sr.','Jr.','Mrs.','Mrs','mrs','Mrs','Mr.','Mr','mr','Dr.','Dr']
    for term in rm_terms:
        try:
            char_names_all.remove(term)
        except:
            pass

    char_names_lower = [x.lower() for x in char_names_all]

    return char_names_all, char_names_lower


def load_place_names():
    proper_names = ['Hogwarts','Gryffindor','Slytherin','Hufflepuff','Ravenclaw',
        'Hallows','Chamber','Hogsmeade','Privet','London']
    settings = [ 'forest', 'castle', 'kitchen', 'dungeon', 'classroom',
        'wood', 'cabin','hallows','office']

    place_names = proper_names + settings

    place_names_lower = [x.lower() for x in place_names]

    return place_names, place_names_lower

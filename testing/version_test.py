from directory_tree import display_tree
customPath = r'C:\Users\Benny\PycharmProjects\gym-flp'
stringRepresentation = display_tree(customPath, string_rep=True, show_hidden=True)
print(stringRepresentation)
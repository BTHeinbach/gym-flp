from directory_tree import display_tree
from bigtree import tree_to_pillow



customPath = r'C:\Users\Benny\PycharmProjects\gym-flp\gym_for_tree_builder'
stringRepresentation = display_tree(customPath, string_rep=True, show_hidden=True)
stringRepresentation= stringRepresentation.replace('gym_for_tree_builder', 'gym-flp')
from bigtree import str_to_tree

tree_str = stringRepresentation
root = str_to_tree(tree_str, tree_prefix_list=["├──", "└──"])
root.show()

pillow_image = tree_to_pillow(root, font_family="arial")
pillow_image.save("tree.jpg")
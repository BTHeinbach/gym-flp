# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:40:56 2022

@author: Shimraz
"""

class Node(object):
    
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []
        self.flat_list = []
 
    def __repr__(self):
        return '<Node name={}>'.format(self.name)
    
    def extract_nodes_containing_string(self, needle, ):
        # TODO: Please implement me
        # needs to be case insensitive
        if self.name.lower() == needle.lower():
            self.flat_list.append(self)
        for child in self.children:
            child.extract_nodes_containing_string(child.name)
        print(self.flat_list)
        return self.flat_list
# Example:
    
def create_tree():
    return Node('root', [
        Node("MaGaZiNo", [
            Node("I"),
            Node("Love"),
            Node("magazino")
        ]),
        Node("Hello", [
            Node("Hello", [
                Node("Hello", [
                    Node("World")
                ])
            ])
        ])
    ])
 
 
root = create_tree()
 
print (root.extract_nodes_containing_string('Hello') == [root])
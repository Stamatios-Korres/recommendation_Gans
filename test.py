class node {

    self.__init__(data):
        self.data = data
        self.next = None

}

class stack{

    def __init__(self):
        self.head = None
        
    def pop(self):
        if head:
            returned_value = head.data
            head = head.next
            return returned_value
        else:
           print('empty_list')

    def push(self,data):
        new_node = node(data)
        new_node.next = self.head
        self.head = new_node
}
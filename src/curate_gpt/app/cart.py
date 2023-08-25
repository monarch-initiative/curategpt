from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class CartItem(BaseModel):
    """
    A cart item is a single item in a cart
    """

    object: Union[Dict, BaseModel]
    object_type: Optional[str] = None


class Cart(BaseModel):
    """
    A cart is a list of items that can be added to or removed from
    """

    items: List[CartItem] = []

    @property
    def size(self):
        return len(self.items)

    def add(self, item: Union[CartItem, Dict]):
        if isinstance(item, dict):
            item = CartItem(object=item)
        self.items.append(item)

    def remove(self, item: CartItem):
        self.items = [i for i in self.items if i != item]

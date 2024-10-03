from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict


class CartItem(BaseModel):
    """
    A cart item is a single item in a cart
    """

    model_config = ConfigDict(protected_namespaces=())
    object: Union[Dict, BaseModel]
    object_type: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict] = None


class Cart(BaseModel):
    """
    A cart is a list of items that can be added to or removed from
    """

    model_config = ConfigDict(protected_namespaces=())
    items: List[CartItem] = []

    @property
    def size(self):
        return len(self.items)

    def add(self, item: Union[CartItem, BaseModel, Dict]):
        if isinstance(item, dict):
            item = CartItem(object=item)
        if not isinstance(item, CartItem):
            if isinstance(item, BaseModel):
                item = CartItem(object=item.dict())
            elif isinstance(item, str):
                item = CartItem(object={"text": item})
            else:
                raise ValueError(f"Invalid cart item: {item}")
        self.items.append(item)

    def remove(self, item: CartItem):
        self.items = [i for i in self.items if i != item]

# from pydantic import BaseModel
# import yaml
#
# class DBConfig(BaseModel):
#     adapter_name: str = "duckdb"
#     ef_construction: int = Field(128, gt=0)
#     ef_search: int = Field(64, gt=0)
#     M: int = Field(16, gt=0)
#     distance_metric: str = "cosine"
#     path: str = Field(default_factory=lambda: default_path(DBConfig.adapter_name))
#     vec_dimension: int = 384
#     default_model: str = "all-MiniLM-L6-v2"
#
#     @validator('ef_construction', 'ef_search', 'M', pre=True, each_item=False)
#     def clamp_values(cls, v, field):
#         max_values = {
#             'ef_construction': 512,
#             'ef_search': 256,
#             'M': 64
#         }
#         min_values = {
#             'ef_construction': 50,
#             'ef_search': 10,
#             'M': 5
#         }
#         if v > max_values[field.name]:
#             return max_values[field.name]
#         elif v < min_values[field.name]:
#             return min_values[field.name]
#         return v
#
#     @validator('path', pre=True, always=True)
#     def set_default_path(cls, v, values):
#         if v is None:
#             return default_path(values.get('adapter_name', 'duckdb'))
#         return v
#
#
# def load_config(config_path: str) -> DBConfig:
#     try:
#         with open(config_path, 'r') as file:
#             config_data = yaml.safe_load(file)
#         return DBConfig(**config_data)
#     except FileNotFoundError:
#         return DBConfig()
#
# def default_path(adapter_name: str) -> str:
#     if adapter_name == "duckdb":
#         return "./duck.db"
#     elif adapter_name == "chromadb":
#         return "./db"
#

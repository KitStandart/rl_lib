
def update_config(config, new_data):
  """Обвновляет конфигурацию по умолчанию"""
  for key, value in new_data.items():
    if isinstance(value, dict) and key in config and isinstance(config[key], dict):
      update_config(config[key], value)
    else:
      config[key] = value

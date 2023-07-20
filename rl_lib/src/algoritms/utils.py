def update_config(config: dict, new_data: dict) -> None:
  """Обвновляет конфигурацию по умолчанию
  Args:
    config: dict: Конфигурация, которую надо обновить
    new_data: dict: Конфигурация с новыми данными
  Returns:
    None
  """
  for key, value in new_data.items():
    if isinstance(value, dict) and key in config and isinstance(config[key], dict):
      update_config(config[key], value)
    else:
      config[key] = value

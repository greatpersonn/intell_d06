import datetime

def str_to_date(date_str: str) -> datetime.date:
    """
    Конвертує рядок у форматі 'YYYY-MM-DD' або 'YYYY/MM/DD' в datetime.date.
    """
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return datetime.datetime.strptime(date_str, "%Y/%m/%d").date()

def is_weekend(date_obj: datetime.date) -> bool:
    """
    Повертає True, якщо date_obj – субота або неділя.
    """
    return date_obj.weekday() >= 5  # 5 = Saturday, 6 = Sunday

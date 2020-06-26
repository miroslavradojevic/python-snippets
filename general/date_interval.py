# get the dates days before the current date
from datetime import date, timedelta

current_date = date.today()
print(type(current_date))
days_before = (current_date - timedelta(days=30)).isoformat()
months_before = (current_date - timedelta(days=270)).isoformat()

print("\ncurrent date: ", current_date.isoformat())
print("30 days before: ", days_before)
print("9 months before: ", months_before)

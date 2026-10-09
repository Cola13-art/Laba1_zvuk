import re # инструмент для работы с регулярными выражениями
import pandas as pd # библиотека для работы с панельными или табличными данными
import pymorphy3 # библиотека для морфологического анализа русских слов
import emoji # библиотека эмодзи
from collections import Counter # импорт класса Counter для ТОП-20

import nltk # библиотека для работы с естественным языком
nltk.pathsec.ALLOW_PROXIED_FETCH = True # переключение флага безопасности, чтобы разрешить nltk использовать системные прокси-соединения при скачивании внешних ресурсов

from nltk.corpus import stopwords # импорт модуля для работы со стоп-словами

nltk.download('stopwords', quiet=True) # скачивание базы стоп-слов

# ============ 1. Загрузка файлов ============
#1 Егор Лонин МТ-203
# Загружаем csv-файл и сохраняем оригинал текста сообщений
df = pd.read_csv('01_internet_magazin.csv', encoding='utf-8-sig')

df['original_text'] = df['message']

# ============ 2. Чистка файлов ============
# Функция первичной очистки, в который удаляются эмодзи, текст переводится в нижний регистр, убираются звездочки, лишние пробелы и повторяющиеся знаки препинания
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = emoji.replace_emoji(text, replace='')
    text = text.lower()
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'!+', '!', text)
    text = re.sub(r'\?+', '?', text)
    text = re.sub(r'\.+', '.', text)
    text = text.strip()
    return text
#Применяем функцию, сохраняя результат в новый столбец
df['basic_clean'] = df['message'].apply(clean_text)

# ============ 3. Поиск сущности через REGEX ============
#2 Данил Констанц МТ-202
# Компиляция регулярных выражений для поиска в тексте сушностей: email, url, телефонные номера, даты и цены
email_re = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
url_re = re.compile(r'(?:https?://|www\.)\S+', re.IGNORECASE)
phone_re = re.compile(r'\+?\d[\d\s\-\(\)]{7,}\d')
date_re = re.compile(r'\b\d{2}[./-]\d{2}[./-]\d{2,4}\b|\b\d{4}[./-]\d{2}[./-]\d{2}\b')
price_re = re.compile(r'\d[\d\s]*\s*(?:руб(?:ль|лей|ля|лях|лем)?|р\.?|rub|₽)', re.IGNORECASE)
#Функция для поиска сущностей с помощью регулярных выражений
def find_all(text):
    #Проверка типа данных
    if not isinstance(text, str):
        return [], [], [], [], []
    #Извлечение сущностей по паттернам
    emails = email_re.findall(text)
    urls_raw = url_re.findall(text)
    dates = date_re.findall(text)
    prices = price_re.findall(text)
    #Даты заменяются на пробелы, чтобы избежать пересечения с поиском номеров
    text_for_phones = date_re.sub(' ', text)
    phones = phone_re.findall(text_for_phones)
    #Очищение ссылок от концевых знаков препинания
    urls = [u.strip().rstrip('.,!?') for u in urls_raw]
    #Очищение цен от лишних внут. пробелов
    prices = [re.sub(r'\s+', ' ', p).strip() for p in prices]
    #Добавление к номерам символа табуляции для корректного отображения в Excel
    phones_clean = ["\t" + p.strip() for p in phones]
    
    return urls, emails, phones_clean, dates, prices
#Создаём списки для каждого типа сущности, в которые раскидываются найденные образцы
urls_list, emails_list, phones_list, dates_list, prices_list = [], [], [], [], []
for text in df['message']:
    u, e, p, d, pr = find_all(text)
    urls_list.append(u)
    emails_list.append(e)
    phones_list.append(p)
    dates_list.append(d)
    prices_list.append(pr)
#Добавляем в таблицу столбцы с данными
df['urls'] = urls_list
df['emails'] = emails_list
df['phones'] = phones_list
df['dates'] = dates_list
df['prices'] = prices_list

# ============ 4. Обезличивание ============

#3 Сергей Таран МТ-203
#Функция для анонимизации, или же замены всех персональных данных на поля по типу [NAME]
def anonymize(text):
    if not isinstance(text, str):
        return ""
    text = email_re.sub('[EMAIL]', text)
    text = url_re.sub('[URL]', text)
    text = date_re.sub('[DATE]', text)
    text = phone_re.sub('[PHONE]', text)
    text = price_re.sub('[PRICE]', text)
    return text
#Добавляем столбец с анонимизированным текстом
df['anonymized_text'] = df['basic_clean'].apply(anonymize)

# ============ 5. Токенизация и стоп-слова ============
#Стоп слова из библиотеки
stop_words = set(stopwords.words('russian'))
#Собственные стоп слова
my_stop = ['это', 'этот', 'весь', 'который', 'свой', 'мой', 'наш', 'ваш',
           'пожалуйста', 'пришлите', 'подскажите', 'ответьте', 'звонить',
           'почта', 'email', 'тел', 'телефон', 'номер', 'ссылка', 'страница',
           'сайт', 'стоимость', 'цена', 'сумма', 'руб', 'рубль', 'рублей',
           'деньги', 'заказ', 'заказа', 'товар', 'товары', 'покупка',
           'магазин', 'сервис', 'проблема', 'вопрос', 'ответ', 'помощь',
           'можно', 'нужно', 'надо', 'очень', 'просто', 'тоже', 'также',
           'уже', 'еще', 'даже', 'ведь', 'вот', 'все', 'какой', 'такой',
           'хочу', 'буду', 'будет', 'был', 'была', 'было', 'были', 'быть',
           'есть', 'нет', 'что', 'делать', 'когда', 'вернутся']
#Обновляем словарь стоп слов, дополняя его своими
stop_words.update(my_stop)
stop_words.discard('не')
#Функция для создания токенов. Удаляет сущности до этого замененные на поля по типу [NAME]. Очищает всё кроме русских и английских букв с пробелами. Разделяет текст на слова и возвращает лишь те слова, у которых длина > 1 символа и которых нет с писке стоп-слов.
def make_tokens(text):
    if not isinstance(text, str):
        return []
    text = re.sub(r'\[(URL|EMAIL|PHONE|DATE|PRICE)\]', ' ', text)
    text = re.sub(r'[^а-яёa-z\s]', ' ', text)
    words = text.split()
    return [w for w in words if len(w) > 1 and w not in stop_words]
#Новый столбец для токенов
df['tokens'] = df['anonymized_text'].apply(make_tokens)

# ============ 6. Лемматизация ============

#4 Станислав Латыпов МТ-203

morph = pymorphy3.MorphAnalyzer()
#Функция, которая приводит каждый токен к словарной форме и отсекает слова длинной в 1 символ.
def lemmatize(tokens):
    lemmas = []
    for t in tokens:
        p = morph.parse(t)
        if p:
            lemma = p[0].normal_form
            if len(lemma) > 1:
                lemmas.append(lemma)
    return lemmas
#Сохраняем в новый столбец для лемм
df['lemmas'] = df['tokens'].apply(lemmatize)
df['clean_text'] = df['lemmas'].apply(lambda x: ' '.join(x))

# ============ 7. ТОП-20 ============

all_words_before = []
for text in df['message']:
    if isinstance(text, str):
        #Поиск слов по шаблону. Фрагменты только из русских и английских букв, с длинной >1 символа
        all_words_before.extend(re.findall(r'[а-яёa-z]{2,}', text.lower()))

#Counter автоматически считает сколько раз каждое слово встречается в тексте
counter_before = Counter(all_words_before)
print("Топ-20 до очистки:")
#most_common подсчитывает наиболее частые
for i, (w, c) in enumerate(counter_before.most_common(20), 1):
    print(f"{i}. {w}: {c}")
print()

all_words_after = []
for lemmas in df['lemmas']:
    #Добавляем леммы
    all_words_after.extend(lemmas)

counter_after = Counter(all_words_after)
print("Топ-20 после очистки:")
for i, (w, c) in enumerate(counter_after.most_common(20), 1):
    print(f"{i}. {w}: {c}")
print()

# ============ 8. Сохранение результата ============
#5 Тимофей Фокин МТ-203
#Разделение даты и времени на два столбца
created_dt = pd.to_datetime(df['created_at'])
df['message_date'] = created_dt.dt.date
df['message_time'] = created_dt.dt.time
#Преобразование списков в строки
df['tokens_str'] = df['tokens'].apply(lambda x: '|'.join(x))
df['lemmas_str'] = df['lemmas'].apply(lambda x: '|'.join(x))
df['emails_str'] = df['emails'].apply(lambda x: '|'.join(x))
df['phones_str'] = df['phones'].apply(lambda x: '|'.join(x))
df['urls_str'] = df['urls'].apply(lambda x: '|'.join(x))
df['dates_str'] = df['dates'].apply(lambda x: '|'.join(x))
df['prices_str'] = df['prices'].apply(lambda x: '|'.join(x))
#Отбираем порядок колонок
result = df[[
    'message_id', 
    'channel', 
    'message_date', 
    'message_time', 
    'clean_text', 
    'anonymized_text', 
    'tokens_str', 
    'lemmas_str', 
    'emails_str', 
    'phones_str', 
    'urls_str', 
    'dates_str', 
    'prices_str'
]]
#Переименуем колонки для удобства
result.columns = [
    'id', 
    'message_type', 
    'date', 
    'time', 
    'clean_text', 
    'anonymized_text', 
    'tokens', 
    'lemmas', 
    'emails', 
    'phones', 
    'urls', 
    'dates', 
    'prices'
]
#Экспорт в csv
result.to_csv('clean_messages.csv', index=False, sep=';', encoding='utf-8-sig')

print("Файл 01_internet_magazin.csv успешно обработан. Файл clean_messages.csv был сохранен")
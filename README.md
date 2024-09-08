# Кейс: Определение зарплатных ожиданий по резюме 

## Команда E-LAN

## Постановка задачи 

На основе двух датасетов («Работа в России»: обработанные и объединенные сведения о вакансиях, резюме, откликах и приглашениях портала trudvsem.ru и Вакансии всех регионов России из ЕЦП «Работа в России» портала trudvsem.ru) создать систему, которая может по текстовому описанию навыков, достижений или образования:

1. Предлагать наиболее предпочтительную
профессию;
2. Предсказывать возможные зарплатные
перспективы для данной специальности по
предлагаемому временному промежутку и
региону.
3. В качестве дополнительной задачи предлагается найти споособ превращения списка должностей в список профессий. 

## Перед запуском кода

- Установленный python 3.12
- Установить необходимые пакеты: `pip install -r requirements.txt`
- В папку data поместить выданные датасеты. Папка data должна выглядеть следующим образом:
```
data
|-JOB_LIST.csv
|-TRAIN_SAL.csv
|-TRAIN_RES_1.csv
|-TRAIN_RES_2.csv
|-TRAIN_RES_3.csv
|-TRAIN_RES_4.csv
|-TRAIN_RES_5.csv
```

## Краткое описание решений

### Задача 1. 

В рамках первой задачи, по своей сути являющейся задачей класссифкации, мы использовали open-source модель rubert-tiny2, к которой на выход добавляем один линейный слой. Для тренировки взяли выборку из датасета с 248 самыми частыми профессиями. Веса rubert не замораживаются, а тоже тренируются. На выходе получаем f1=0.37 на тестовой выборке.

Ноутбук с кодом лежит [тут](job_name_prediction.ipynb)

### Задача 2.

При решении второй задачи -- задачи регрессии мы делалил ... 

Ноутбук с кодом лежит [тут]()

### Задача 3. 

Решение третьей задачи строится на получении эмбеддингов должностей с помощью обученной нейронной сети, затем применяется кластеризация методом DBSCAN, так как точное количество кластеров заранее неизвестно и этот метод хорошо работает с неравномерными данными.

Первичный результат был хорошим на глаз (к сожалению, придумать метрику качесвто не удалось), но для улучшения предложены следующие шаги:

1. Итеративный DBSCAN на усредненных векторах найленных классов и шумовых данных.
2. Использование метода ближайших соседей для итеративного поиска ближайшего соседа для каждого элемента из класса шума. 
_Данный способ кажется нам наиболее удачным_. Для достижения наилучших результатов стоит еще раз воспользоваться методом класстеризации DBSCAN для поиска схожих классов и объединения их в один класс. 
3. При необходимости, тщательный подбор параметров первичной кластеризации или использование другой модели для получения эмбеддингов.

Для определения названия классов предлагается использовать самое короткое название должности в пределах каждого класса. При корректной кластеризации это позволит точно определить название класса, а следовательно, и соответствующей профессии.

Ноутбук с кодом лежит [тут](position_2_occupation.ipynb)
# -*- coding: utf-8 -*-
"""
ПО «Скрининг повышения НУП по ИИ-ЭКГ»
Расчёт вероятности повышения уровня натрийуретического пептида (NT-proBNP)
выше возрастной нормы по 47 параметрам одноканальной ЭКГ.

Версия: 1.0.0
Разработчик: Институт персонализированной кардиологии,
             Сеченовский Университет

Порядок работы:
  1. Программа запрашивает путь к файлу с данными (.xlsx).
  2. Программа запрашивает путь к файлу модели (model_bnp.json).
  3. Для каждой записи рассчитывается вероятность повышения NT-proBNP (0..1)
     и заключение по порогу 0,55.
  4. Результаты записываются в файл bnp_test_results.txt
     в папке с файлом данных.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

VERSION = '1.0.0'
THRESHOLD = 0.55
RESULT_FILE = 'bnp_test_results.txt'
ID_COLUMN = 'ID записи'

# 47 параметров одноканальной ЭКГ в порядке, на котором обучена модель
FEATURES = [
    'HFNoise', 'RR', 'TpTe', 'VAT', 'QTc', 'QT/TQ', 'HFQRS', 'HFSNR', 'JA', 'J80A',
    'TA', 'QRSenergy', 'Tenergy', 'Tpenergy', 'Sbeta', 'Beta', 'QRS11energy',
    'QRS12energy', 'QRS2energy', 'QRSE1', 'QRSE2', 'QRSE3', 'QRSE4', 'TE1', 'TE2',
    'TE3', 'TE4', 'QRSw', 'Pan', 'Pan.1', 'RA', 'SA', 'Pst', 'Pfi', 'QRSst',
    'QRSfi', 'Tfi', 'PpeakP', 'PpeakN', 'Rpeak', 'Speak', 'Tpeak', 'Tons', 'Toffs',
    'RonsF', 'RoffsF', 'SDNN',
]


class InputError(Exception):
    """Ошибка входных данных, понятная пользователю."""


def ask_path(prompt):
    """Запрашивает путь у пользователя и убирает лишние кавычки и пробелы."""
    return input(prompt).strip().strip('"').strip("'")


def check_path(path, extension):
    if not path or not os.path.isfile(path):
        raise InputError(
            'Ошибка при вводе пути!\n'
            'Неправильно введён путь к данным, либо к Системе выявления '
            'повышения уровня НУП в крови.\n'
            'Проверьте правильность ввода пути.')
    if not path.lower().endswith(extension):
        raise InputError(
            f'Ошибка формата файла!\n'
            f'Ожидается файл с расширением {extension}, получен: '
            f'{os.path.basename(path)}')


def load_data(path):
    """Читает xlsx-файл: одна строка = одна запись ЭКГ, 47 параметров в столбцах."""
    check_path(path, '.xlsx')
    try:
        df = pd.read_excel(path)
    except Exception as exc:
        raise InputError(f'Ошибка чтения файла данных: {exc}')
    if df.empty:
        raise InputError('Ошибка входных данных: файл не содержит записей.')

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise InputError(
            'Ошибка входных данных: в файле отсутствуют обязательные параметры ЭКГ:\n  '
            + ', '.join(missing))

    x = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    bad_rows = x.index[x.isna().any(axis=1)].tolist()
    if bad_rows:
        rows = ', '.join(str(i + 2) for i in bad_rows[:10])
        raise InputError(
            'Ошибка входных данных: пустые или нечисловые значения параметров ЭКГ '
            f'в строках файла: {rows}' + (' …' if len(bad_rows) > 10 else ''))

    ids = (df[ID_COLUMN].astype(str) if ID_COLUMN in df.columns
           else pd.Series([str(i + 1) for i in range(len(df))]))
    return ids.tolist(), x.astype(float)


def load_model(path):
    check_path(path, '.json')
    booster = xgb.Booster()
    try:
        booster.load_model(path)
    except Exception as exc:
        raise InputError(f'Ошибка загрузки Системы (файла модели): {exc}')
    if booster.feature_names and list(booster.feature_names) != FEATURES:
        raise InputError('Ошибка: файл модели не соответствует данной версии ПО.')
    return booster


def predict(booster, x):
    dm = xgb.DMatrix(x.values, feature_names=FEATURES)
    return booster.predict(dm)


def save_results(ids, proba, data_path):
    out_path = os.path.join(os.path.dirname(os.path.abspath(data_path)), RESULT_FILE)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f'ПО «Скрининг повышения НУП по ИИ-ЭКГ», версия {VERSION}\n')
        f.write(f'Файл данных: {os.path.basename(data_path)}\n')
        f.write(f'Дата расчёта: {time.strftime("%d.%m.%Y %H:%M:%S")}\n')
        f.write(f'Пороговое значение вероятности: {THRESHOLD:.2f}\n\n')
        f.write('ID записи\tВероятность\tЗаключение\n')
        for i, p in zip(ids, proba):
            verdict = ('Повышение НУП вероятно' if p >= THRESHOLD
                       else 'Повышение НУП маловероятно')
            f.write(f'{i}\t{p:.4f}\t{verdict}\n')
    return out_path


def main():
    print(f'ПО «Скрининг повышения НУП по ИИ-ЭКГ», версия {VERSION}')
    try:
        data_path = ask_path('Введите полный путь к данным: ')
        ids, x = load_data(data_path)
        model_path = ask_path(
            'Введите полный путь к Системе выявления повышения НУП по ИИ-ЭКГ: ')
        booster = load_model(model_path)

        t0 = time.perf_counter()
        proba = predict(booster, x)
        elapsed = time.perf_counter() - t0

        out_path = save_results(ids, proba, data_path)
    except InputError as err:
        print(err)
        return 1
    except KeyboardInterrupt:
        print('\nРабота прервана пользователем.')
        return 1

    print(f'Обработано записей: {len(proba)}')
    print(f'Время расчёта: {elapsed:.3f} с')
    print(f'Порог: {THRESHOLD:.2f}; повышение НУП вероятно в {int((proba >= THRESHOLD).sum())} '
          f'из {len(proba)} записей')
    print(f'Результаты записаны в файл: {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

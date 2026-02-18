import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import tensorflow as tf
from tensorflow import keras
from pdf_generator import export_to_pdf

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="VetAI v0.4.1 (Alpha): Протокол Диагностики",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="auto"
)

# Заголовок страницы
st.title('🩺 VetAI v0.4.1 (Alpha): Ассистент Диагностики')
st.markdown('### Клинически структурированный протокол сбора данных')
st.markdown('---')

# --- 1. Загрузка компонентов (РЕАЛЬНЫЕ КОМПОНЕНТЫ v15) ---
MODEL_VERSION = 'v15 (150 классов)'
MODEL_PATH = 'full_neural_network_model_v15_opt.h5'
PREPROCESSOR_PATH = 'full_preprocessor_v15.pkl'
ENCODER_PATH = 'full_label_encoder_v15.pkl'
FEATURES_PATH = 'full_feature_names_v15.pkl' 

@st.cache_resource
def load_artifacts():
    try:
        # Модель Keras
        model = keras.models.load_model(MODEL_PATH)
        # Препроцессор (StandardScaler)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        # Энкодер меток
        label_encoder = joblib.load(ENCODER_PATH)
        # Список всех признаков (критично для выравнивания)
        training_features = joblib.load(FEATURES_PATH) 
        
        st.success(f"Модель **VetAI v{MODEL_VERSION}** ({len(training_features)} признаков) успешно загружена и готова к работе.")
        return model, preprocessor, label_encoder, training_features
    except Exception as e:
        st.error(f"⚠️ Ошибка загрузки артефактов модели v15: {e}. Используются заглушки.")
        
        # --- ЗАГЛУШКИ (Используются для корректной работы форматирования) ---
        class DummyModel:
            def predict(self, data):
                # Возвращаем примеры ключей диагнозов, которые есть в словаре-переводчике
                return np.array([[0.65, 0.25, 0.10]]) 
        class DummyPreprocessor:
            def transform(self, data):
                return np.zeros((1, 3)) 
        class DummyEncoder:
            def inverse_transform(self, indices):
                # Имитируем вывод ключей диагнозов
                classes = ['гастрит_эозинофильный', 'пироплазмоз_осложненный', 'панкреатит_тяжелый']
                return [classes[i] for i in indices]
        
        return DummyModel(), DummyPreprocessor(), DummyEncoder(), [] 

model, preprocessor, label_encoder, TRAINING_FEATURES = load_artifacts()

if not TRAINING_FEATURES and not isinstance(model, DummyModel):
    st.error("Критическая ошибка: список признаков не загружен. Невозможно выровнять входные данные.")
    st.stop()


# --- СЛОВАРЬ ДЛЯ ПЕРЕВОДА ДИАГНОЗОВ (КЛЮЧИ ИЗ МОДЕЛИ -> РУССКИЙ + ЛАТЫНЬ) ---
# ВНИМАНИЕ: ДОБАВЬТЕ СЮДА ВСЕ 149 ДИАГНОЗОВ В ФОРМАТЕ 'ключ_из_модели': 'Русское название (Латинское название)'
DIAGNOSIS_TRANSLATIONS = {
    'гастрит_эозинофильный': 'Эозинофильный гастрит (Gastritis eosinophilica)',
    'панкреатит_тяжелый': 'Тяжелый панкреатит (Pancreatitis gravis)',
    'пироплазмоз_осложненный': 'Осложненный пироплазмоз (Piroplasmosis complicata)',
    'энтерит_парвовирусный': 'Парвовирусный энтерит (Enteritis parvovirosa)',
    'холецистит': 'Холецистит (Cholecystitis)',
    'отравление_антифризом': 'Отравление антифризом (Intoxicatio antigelum)',
    'недостаточность_почечная_острая': 'Острая почечная недостаточность (Insuff. renalis acuta)',
    'аллергический_дерматит_пищевой': 'Пищевой аллергический дерматит (Dermatitis allergica alimentaria)',
    # ДОБАВЬТЕ ОСТАЛЬНЫЕ КЛЮЧИ ВАШЕЙ МОДЕЛИ СЮДА
}


# --- 2. Определение и группировка признаков (UI и DataFrame) ---

# ДЕМОГРАФИЯ (Ключи для OHE)
demographic_keys = {
    'Порода': {'собака': 'Собака', 'кошка': 'Кошка'},
    'Возраст': {'новорожденный': 'Новорожденный (до 1 мес.)', 'щенок_котенок': 'Молодняк', 'молодой': 'Молодой (1-5 лет)', 'взрослый': 'Взрослый (6-9 лет)', 'старый': 'Старый (10+ лет)'},
    'Пол': {'самец': 'Самец', 'самка': 'Самка'},
    'Вес': {'низкий': 'Низкий, истощение', 'нормальный': 'Нормальный', 'высокий': 'Высокий, лишний вес', 'ожирение_сильное': 'Ожирение, сильное'},
    'Температура': {'нормальная': 'Нормальная', 'повышенная': 'Повышенная', 'критично_повышенная': 'Критично повышенная', 'гипотермия': 'Пониженная'},
    'Содержание': {'дома': 'Дома (квартирное)', 'улица': 'На улице', 'частный_дом': 'Частный дом'},
    'Диета': {'корм': 'Готовый корм', 'натуралка': 'Натуральная пища'},
    # Vitals (для OHE)
    'Общее_состояние': {'нормальное': 'Нормальное', 'средней_тяжести': 'Средней тяжести', 'тяжелое': 'Тяжелое', 'критическое': 'Критическое'},
    'Ментальный_статус': {'bar': 'BAR', 'qar': 'QAR', 'депрессия': 'Депрессия', 'ступор': 'Ступор', 'кома': 'Кома'},
    'Пульс': {'нормальный': 'Нормальный', 'тахикардия': 'Тахикардия', 'брадикардия': 'Брадикардия', 'аритмия': 'Аритмия', 'слабый': 'Слабый', 'напряженный': 'Напряженный'},
    'Аускультация_сердца': {'шумов_нет': 'Шумов нет', 'систолические_шумы': 'Систолические шумы', 'диастолические_шумы': 'Диастолические шумы', 'глухие_тоны': 'Глухие тоны'},
    'Дыхание_паттерн': {'норма': 'Нормальная частота и глубина', 'тахипноэ': 'Тахипноэ', 'брадипноэ': 'Брадипноэ', 'гиперпноэ': 'Гиперпноэ', 'чейна_стокса': 'Дыхание Чейна-Стокса', 'биота': 'Дыхание Биота', 'куссмауля': 'Дыхание Куссмауля', 'грокко': 'Дыхание Грокко', 'черни': 'Дыхание Черни', 'диспноэ': 'Диспноэ', 'поверхностное': 'Поверхностное', 'критически_редкое': 'Критически редкое'},
    'Аускультация_легких': {'везикулярное': 'Везикулярное (Норма)', 'хрипы_сухие': 'Хрипы сухие', 'хрипы_влажные': 'Хрипы влажные', 'бронхиальное': 'Бронхиальное', 'крепитация': 'Крепитация', 'жесткое_бронхиальное': 'Жесткое бронхиальное', 'ослабленное': 'Ослабленное'},
    'Тип_дыхания': {'грудо_брюшное': 'Грудо-брюшное', 'брюшное': 'Брюшное', 'смешанное': 'Смешанное (Норма)'},
    'Дегидратация': {'нет': 'Нет', '1_стадия': '1 стадия (до 5%)', '2_стадия': '2 стадия (5-6%)', '3_стадия': '3 стадия (7-8%)', 'тяжелый_шок': 'Тяжелый шок (8+%)'},
    'КНС': {'норма': 'Норма (1-2 сек)', 'замедленное': 'Замедленное (КНС > 3 сек)', 'ускоренное': 'Ускоренное (КНС < 1 сек)'},
    'Тонометрия': {'не_проводилась': 'Не проводилась', 'нормальное': 'Нормальное', 'повышенное': 'Повышенное', 'пониженное': 'Пониженное'}
}

# ВАКЦИНЫ (Ключи для OHE)
VACCINES_DOGS_VIRUS = {'Нет_вирусной_вакцинации': 'Нет', 'biocan_dhppi': 'Биокан DHPPI', 'multican_8': 'Мультикан-8', 'multican_4_6': 'Мультикан-4/6', 'eurican': 'Эурикан', 'asterion': 'Астерион', 'hexacanivac': 'Гексаканивак', 'nobivac_dhppi': 'Нобивак DHPPI', 'nobivac_l4': 'Нобивак L4',}
VACCINES_DOGS_RABIES = {'Нет_бешенства': 'Нет', 'rabisin_d': 'Рабизин', 'rabikan': 'Рабикан', 'nobivac_rabies': 'Нобивак Rabies', 'defensor': 'Дефенсор', 'nobivac_rl': 'Нобивак RL'} 
DOG_COMPLEX_RABIES = {'Мультикан-8': 'Мультикан-8 (комплексная)', 'Нобивак RL': 'Нобивак RL (Бешенство + Лептоспироз)'}

VACCINES_CATS_VIRUS = {'Нет_вирусной_вакцинации': 'Нет', 'multifel_4': 'Мультифел 4', 'purevax_3': 'Пуревакс RCP', 'purevax_4': 'Пуревакс RCPCh', 'purevax_felv': 'Пуревакс FeLV', 'nobivac_triket': 'Нобивак Трикет', 'biofel_pch': 'Биофел PCH', 'leucogen': 'Лейкоген',}
VACCINES_CATS_RABIES = {'Нет_бешенства': 'Нет', 'rabifel': 'Рабифел', 'rabisin_c': 'Рабизин', 'nobivac_rabies': 'Нобивак Rabies', 'defensor': 'Дефенсор', 'biofel_pchr': 'Биофел PCHR', 'quadricat': 'Квадрикат'}
CAT_COMPLEX_RABIES = {'Мультифел 4': 'Мультифел 4 (комплексная)', 'Биофел PCHR': 'Биофел PCHR (комплексная)', 'Квадрикат': 'Квадрикат (комплексная)'}


# --- КЛИНИЧЕСКИЕ СИМПТОМЫ (Бинарные ключи) ---
symptom_groups = {
    "Общие, Системные": [
        ('Гиподинамия', 'Гиподинамия (снижение активности)'), ('Летаргия', 'Летаргия (ступор)'), 
        ('Анорексия', 'Анорексия (Полный отказ от еды)'), ('Гипорексия', 'Гипорексия (Снижение аппетита)'),
        ('Избирательный_аппетит', 'Избирательный аппетит'), ('Пикоцизм', 'Пикоцизм'), 
        ('Тошнота', 'Тошнота'), ('Борборигмы', 'Борборигмы'),
        ('Потеря_веса', 'Заметное похудение'), ('Дрожь_тела', 'Дрожь, тремор тела'),
        ('Шок_слабость', 'Состояние шока'), ('Не_дает_прикоснуться', 'Боль, не дает к себе прикоснуться'),
        ('Вокализация', 'Ненормальная вокализация'), ('Агрессия', 'Агрессия'),
        ('Прячется_изолируется', 'Прячется, изолируется'), ('Без_сознания', 'Потеря сознания')
    ],
    "Пищеварение (ЖКТ, Рот)": [
        ('Дисфагия', 'Дисфагия'), ('Регургитация', 'Регургитация'),
        ('Рвота_тащековая', 'Рвота тащековая'), ('Рвота_постбрандиальная_рано', 'Рвота постбрандиальная (до 30 мин)'),
        ('Рвота_постбрандиальная_поздно', 'Рвота постбрандиальная (1-2 часа)'), ('Рвота_смешанная', 'Рвота смешанная'),
        ('Полидипсия', 'Полидипсия'), ('Олигодипсия', 'Олигодипсия'), ('Адипсия', 'Адипсия'),
        ('Толстокишечная_диарея', 'Диарея (Толстокишечная)'), ('Тонкокишечная_диарея', 'Диарея (Тонкокишечная)'), ('Смешанная_диарея', 'Диарея (Смешанная)'), 
        ('Запор', 'Запор'), ('Следы_крови_алая', 'Кровь в стуле (алая)'), ('Следы_крови_спекшаяся', 'Кровь в стуле (спекшаяся)'),
        ('Стеатоарея', 'Стеатоарея (слизь в кале)'), ('Кал_белый_желтый', 'Кал необычного цвета (белый/желтый)'), ('Мелена', 'Мелена (черный кал)'),
        ('Яйца_глистов_в_кале', 'Видимые яйца глистов'),
        ('Живот_увеличен', 'Живот увеличен'), ('Живот_болезнен', 'Живот болезненный'), ('Живот_напряжен', 'Живот напряжен'),
        ('Неприятный_запах_изо_рта', 'Неприятный запах изо рта'), ('Зубной_камень', 'Обильный зубной камень'), 
        ('Генгивит_эрозии_язвы', 'Генгивит/Стоматит, Эрозии/Язвы'), ('Гиперсаливация', 'Гиперсаливация'),
        ('Птеолизм', 'Птиализм'), ('Патологии_челюсти_зубов', 'Патологии зубов/челюсти')
    ],
    "Лимфоузлы и Пальпация": [
        ('Увеличение_нижнечелюстных_ЛУ', 'Увеличение нижнечелюстных ЛУ'), ('Увеличение_подмышечных_ЛУ', 'Увеличение подмышечных ЛУ'), 
        ('Увеличение_паховых_ЛУ', 'Увеличение паховых ЛУ'), ('Увеличение_подколенных_ЛУ', 'Увеличение подколенных ЛУ'), 
        ('Увеличенные_почки', 'Увеличенные почки'), ('Неровные_контуры_почек', 'Неровные контуры почек'),
        ('Обнаружена_дилатация', 'Обнаружена дилатация'),
        ('Болезненность_мочевого_пузыря', 'Мочевой пузырь болезнен/напряжен'), ('Патология_ректального_осмотра', 'Патология при ректальном осмотре')
    ],
    "Дыхательная система": [
        ('Кашель_сухой', 'Кашель (сухой)'), ('Кашель_влажный', 'Кашель (влажный)'), ('Чихание', 'Частое чихание'), 
        ('Ринорея_простая', 'Ринорея (простая)'), ('Ринорея_гнойно_катаральная', 'Ринорея (гнойно-катаральная)'), 
        ('Свист_при_дыхании', 'Свист, хрипы при дыхании'), 
        ('Тест_холодного_стекла_отриц_одно', 'Тест холодного стекла (-) односторонний'),
        ('Тест_холодного_стекла_отриц_дву', 'Тест холодного стекла (-) двусторонний'),
        ('Увеличение_носовых_ходов', 'Увеличение носовых ходов'),
    ],
    "Мочеполовая система": [
        ('Полнуюрия', 'Полнуюрия'),
        ('Поллакиурия', 'Поллакиурия'), 
        ('Олигурия', 'Олигурия'), 
        ('Анурия', 'Анурия'), 
        ('Дизурия_странгурия', 'Дизурия/Странгурия'), 
        ('Ишурия', 'Ишурия'),
        ('Периурия', 'Периурия'),
        ('Недержание_мочи', 'Неконтролируемое мочеиспускание'), 
        ('Гематурия', 'Гематурия'), 
        ('Изменение_запаха_мочи', 'Изменение запаха мочи'), ('Моча_необычного_цвета', 'Моча необычного цвета'),
        ('Выделения_из_пениса', 'Выделения из препуция/пениса'), ('Выделения_из_петли', 'Выделения из петли/влагалища'), 
        ('Гиперемия_пениса', 'Гиперемия пениса'), ('Агенизия', 'Агенизия'),
        ('Воспаление_молочных_желез', 'Воспаление молочных желез'),
    ],
    "Нервная система": [
        ('Атоксия', 'Атоксия'),
        ('Судороги_клонические', 'Судороги (клонические)'), ('Судороги_тонические', 'Судороги (тонические)'), ('Судороги_клоникотонические', 'Судороги (клоникотонические)'),
        ('Дискенизия', 'Дискенизия'), ('Мозжечковая_патология', 'Мозжечковая патология'), 
        ('Вестибулярная_патология', 'Вестибулярная патология'), ('Проприоцептивная_патология', 'Проприоцептивная патология'),
        ('Отсутствие_ГБЧ', 'Отсутствие ГБЧ'), ('Сниженная_проприоцепция', 'Сниженная проприоцепция'),
        ('Аллодиния', 'Аллодиния'), ('Хромота', 'Хромота/боль при движении'), 
    ],
    "Кожа, Шерсть": [
        ('Зуд_чешется', 'Сильный зуд'), ('Алопеция', 'Алопеция'), ('Лехинефикация', 'Лихенификация'), 
        ('Покраснение_кожи', 'Покраснение'), ('Сыпь_пузырьки', 'Сыпь, водянистые пузырьки'), 
        ('Перхоть_шелушение', 'Перхоть, шелушение'), ('Ранки_на_коже', 'Множественные ранки'), 
        ('Уплотнения_шишки', 'Уплотнения, шишки'), ('Катание_на_попе', 'Катание на попе'), 
        ('Изменение_цвета_шерсти', 'Изменение цвета шерсти'), ('Лампас_на_спине_у_кошек', 'Лампас на спине'),
        ('Выпадение_шерсти', 'Выпадение шерсти')
    ],
    "Глаза, Уши, Нос": [
        ('Эпифора_гнойная', 'Эпифора (гнойная)'), ('Эпифора_гнойно_катаральная', 'Эпифора (гнойно-катаральная)'), ('Эпифора_простая', 'Эпифора (простая)'),
        ('Эпифора_односторонняя', 'Эпифора (одностороннее)'), ('Онизокария', 'Онизокария'), 
        ('Энофтальм', 'Энофтальм'), ('Экзофтальм', 'Экзофтальм'), 
        ('Отсутствие_реакции_на_свет', 'Отсутствие реакции на свет'), ('Выпадение_третьего_века_дву', 'Выпадение третьего века (двустороннее)'), 
        ('Выпадение_третьего_века_одно', 'Выпадение третьего века (одностороннее)'), ('Стробизм_дивергентный', 'Стробизм (дивергентный)'), 
        ('Стробизм_конвергентный', 'Стробизм (конвергентный)'), ('Нистагм', 'Нистагм'),
        ('Гиперемия_глаз', 'Гиперемия глаз'), ('Отечность_глаз', 'Отечность глаз'),
        ('Гиперемия_ушей', 'Гиперемия ушей'), ('Коричневые_выделения_ушей_одно', 'Коричневые выделения ушей (одностороннее)'),
        ('Коричневые_выделения_ушей_дву', 'Коричневые выделения ушей (двустороннее)'), ('Отечность_ушей_одно', 'Отечность ушей (одностороннее)'),
        ('Отечность_ушей_дву', 'Отечность ушей (двустороннее)'),
    ],
    "Травмы и Кровотечения": [
        ('Кровотечение_артериальное', 'Кровотечение (артериальное)'), ('Кровотечение_венозное', 'Кровотечение (венозное)'),
        ('Кровотечение_капиллярное', 'Кровотечение (капиллярное)'), ('Петехии', 'Петехии'),
        ('Гематома', 'Гематома'), ('Серома', 'Серома'), ('Высотная_травма', 'Высотная травма'),
        ('Краш_синдром', 'Краш-синдром'), ('Электротравма', 'Электротравма'),
        ('Открытые_раны', 'Открытые раны'), ('Ушибы_синяки', 'Обширные ушибы'), 
        ('Признаки_переломов', 'Подозрение на перелом'), ('Ожоги', 'Ожоги'), ('Обморожения', 'Обморожения'),
        ('Кусал_клещ', 'Факт: кусал клещ'), ('Следы_крови_на_подстилке', 'Следы крови на подстилке')
    ]
}


# --- 3. ИНТЕРФЕЙС: РАЗДЕЛЕНИЕ НА ВКЛАДКИ (По протоколу Тайсона) ---

tab_anamnesis, tab_vitals, tab_symptoms = st.tabs([
    "1. Анамнез & История", 
    "2. Клиническое обследование", 
    "3. Клинические симптомы (Жалобы)"
])

# --- 3.1. АНАМНЕЗ & ИСТОРИЯ ---
with tab_anamnesis:
    st.header('Общие данные и Анамнез')
    
    # --- Демография ---
    col_breed, col_age, col_gender = st.columns(3)
    with col_breed:
        breed_label = st.selectbox('Вид животного', list(demographic_keys['Порода'].values()), key='breed_label')
        breed_key = [k for k, v in demographic_keys['Порода'].items() if v == breed_label][0]
    with col_age:
        age_label_selected = st.selectbox('Возрастная группа', list(demographic_keys['Возраст'].values()), key='age_label')
        age_key = [k for k, v in demographic_keys['Возраст'].items() if v == age_label_selected][0]
    with col_gender:
        gender_label_selected = st.selectbox('Пол', list(demographic_keys['Пол'].values()), key='gender_label')
        gender_key = [k for k, v in demographic_keys['Пол'].items() if v == gender_label_selected][0]
    
    st.markdown('---')
    col_castr, col_own = st.columns(2)
    with col_castr:
        is_castrated = st.checkbox('Проведена **Кастрация** (для самцов/самок)', key='is_castrated')
    with col_own:
        ownership_duration = st.selectbox('Длительность владения', ['Недавно', 'Менее года', 'Более года'], key='ownership_duration')

    # --- Условия и история ---
    st.subheader('Условия содержания и история')
    col_housing, col_diet = st.columns(2)
    with col_housing:
        housing_options = ['Комбинированное'] + list(demographic_keys['Содержание'].values())
        housing_label = st.selectbox('Условия содержания', housing_options, key='housing_label')
    with col_diet:
        diet_options = ['Комбинированное'] + list(demographic_keys['Диета'].values())
        diet_label = st.selectbox('Тип кормления', diet_options, key='diet_label')
        
    col_hist_1, col_hist_2, col_hist_3, col_hist_4 = st.columns(4)
    with col_hist_1:
        has_chronic = st.checkbox('Есть хронические заболевания', key='has_chronic')
    with col_hist_2:
        has_surgery = st.checkbox('Перенесены операции/заболевания', key='has_surgery')
    with col_hist_3:
        has_antibio = st.checkbox('Был антибиотикоанамнез', key='has_antibio')
    with col_hist_4:
        allergy_status = st.checkbox('Есть аллергия', key='allergy_status')
        
    st.markdown('---')
    
    # --- Профилактика ---
    st.subheader('Профилактика и Вакцинация')
    col_vax_status, col_deworm, col_ectopara, col_cohabiting = st.columns(4)
    with col_vax_status:
        vax_status = st.selectbox('Проведена вакцинация?', ['Нет', 'Да'], key='vax_status')
    with col_deworm:
        deworm_status = st.selectbox('Проведена дегельминтизация?', ['Нет', 'Да'], key='deworm_status')
    with col_ectopara:
        ectopara_status = st.selectbox('Обработка от блох/клещей?', ['Нет', 'Да'], key='ectopara_status')
    with col_cohabiting:
        cohabiting_status = st.selectbox('Есть ли еще животные?', ['Нет', 'Да, здоровы', 'Да, с симптомами'], key='cohabiting_status')

    # Инициализация ключей вакцин
    vaccine_selection_virus_key = 'Нет_вирусной_вакцинации'
    vaccine_selection_rabies_key = 'Нет_бешенства'
    # Инициализация для устранения NameError
    is_complex_rabies_dog = False 
    is_complex_rabies_cat = False
    
    if vax_status == 'Да':
        st.markdown('###### Выберите проведенную вакцину (только ОДНУ):')
        col_vax_v, col_vax_r = st.columns(2)
        
        if breed_key == 'собака':
            virus_options = list(VACCINES_DOGS_VIRUS.values())
            complex_options = list(DOG_COMPLEX_RABIES.values())
            all_virus_options = ['Нет'] + virus_options + complex_options
            
            with col_vax_v:
                vax_virus_label = st.selectbox('Вирусная / Комплексная вакцина', options=all_virus_options, key='vax_virus_select')
                if vax_virus_label == 'Нет':
                       vaccine_selection_virus_key = 'Нет_вирусной_вакцинации'
                else:
                    # ИСПРАВЛЕНИЕ: Используем {**A, **B} вместо A | B для Python 3.8
                    vaccine_selection_virus_key = [k for k, v in {**VACCINES_DOGS_VIRUS, **DOG_COMPLEX_RABIES}.items() if v == vax_virus_label][0]

            is_complex_rabies_dog = vax_virus_label in complex_options
            
            with col_vax_r:
                if is_complex_rabies_dog:
                    st.selectbox('От Бешенства (Моновакцина)', options=[f"{vax_virus_label} (Комплексная)"], disabled=True, key='vax_rabies_select_disabled')
                    if vaccine_selection_virus_key == 'nobivac_rl':
                         vaccine_selection_rabies_key = 'nobivac_rl'
                    else:
                         vaccine_selection_rabies_key = 'rabies_included_in_complex' 
                else:
                    rabies_options = ['Нет'] + list(VACCINES_DOGS_RABIES.values())
                    vax_rabies_label = st.selectbox('От Бешенства (Моновакцина)', options=rabies_options, key='vax_rabies_select')
                    if vax_rabies_label == 'Нет':
                        vaccine_selection_rabies_key = 'Нет_бешенства'
                    else:
                        vaccine_selection_rabies_key = [k for k, v in VACCINES_DOGS_RABIES.items() if v == vax_rabies_label][0]


        elif breed_key == 'кошка':
            virus_options = list(VACCINES_CATS_VIRUS.values())
            complex_options = list(CAT_COMPLEX_RABIES.values())
            all_virus_options = ['Нет'] + virus_options + complex_options
            
            with col_vax_v:
                vax_virus_label = st.selectbox('Вирусная / Комплексная вакцина', options=all_virus_options, key='vax_virus_select_c')
                if vax_virus_label == 'Нет':
                       vaccine_selection_virus_key = 'Нет_вирусной_вакцинации'
                else:
                    # ИСПРАВЛЕНИЕ: Используем {**A, **B} вместо A | B для Python 3.8
                    vaccine_selection_virus_key = [k for k, v in {**VACCINES_CATS_VIRUS, **CAT_COMPLEX_RABIES}.items() if v == vax_virus_label][0]

            is_complex_rabies_cat = vax_virus_label in complex_options
            
            with col_vax_r:
                if is_complex_rabies_cat:
                    st.selectbox('От Бешенства (Моновакцина)', options=[f"{vax_virus_label} (Комплексная)"], disabled=True, key='vax_rabies_select_disabled_c')
                    vaccine_selection_rabies_key = 'rabies_included_in_complex'
                else:
                    rabies_options = ['Нет'] + list(VACCINES_CATS_RABIES.values())
                    vax_rabies_label = st.selectbox('От Бешенства (Моновакцина)', options=rabies_options, key='vax_rabies_select_c')
                    if vax_rabies_label == 'Нет':
                        vaccine_selection_rabies_key = 'Нет_бешенства'
                    else:
                        vaccine_selection_rabies_key = [k for k, v in VACCINES_CATS_RABIES.items() if v == vax_rabies_label][0]


# --- 3.2. КЛИНИЧЕСКОЕ ОБСЛЕДОВАНИЕ (VITALS) ---
with tab_vitals:
    st.header('Объективные показатели и Обследование')
    
    # --- Общие показатели ---
    col_overall, col_mental, col_weight = st.columns(3)
    with col_overall:
        overall_status_label = st.selectbox('Общее состояние', list(demographic_keys['Общее_состояние'].values()), key='overall_status')
        overall_status_key = [k for k, v in demographic_keys['Общее_состояние'].items() if v == overall_status_label][0]
    with col_mental:
        mental_status_label = st.selectbox('Ментальный статус', list(demographic_keys['Ментальный_статус'].values()), key='mental_status')
        mental_status_key = [k for k, v in demographic_keys['Ментальный_статус'].items() if v == mental_status_label][0]
    with col_weight:
        weight_label = st.selectbox('Вес и кондиция', list(demographic_keys['Вес'].values()), key='weight_label_vitals')
        weight_key_vitals = [k for k, v in demographic_keys['Вес'].items() if v == weight_label][0] 

    col_temp, col_kns, col_dehydr = st.columns(3)
    with col_temp:
        temp_label = st.selectbox('Температура тела', list(demographic_keys['Температура'].values()), key='temp_label_vitals')
        temp_key_vitals = [k for k, v in demographic_keys['Температура'].items() if v == temp_label][0]
    with col_kns:
        kns_status_label = st.selectbox('КНС', list(demographic_keys['КНС'].values()), key='kns_status')
        kns_status_key = [k for k, v in demographic_keys['КНС'].items() if v == kns_status_label][0]
    with col_dehydr:
        dehydration_status_label = st.selectbox('Признаки дегидратации', list(demographic_keys['Дегидратация'].values()), key='dehydration_status')
        dehydration_status_key = [k for k, v in demographic_keys['Дегидратация'].items() if v == dehydration_status_label][0]
        
    st.subheader('Сердечно-дыхательная система')
    col_pulse, col_heart_ausc, col_resp_rate = st.columns(3)
    with col_pulse:
        pulse_status_label = st.selectbox('Пульс (Частота и качество)', list(demographic_keys['Пульс'].values()), key='pulse_status')
        pulse_status_key = [k for k, v in demographic_keys['Пульс'].items() if v == pulse_status_label][0]
    with col_heart_ausc:
        auscultation_heart_label = st.selectbox('Аускультация сердца', list(demographic_keys['Аускультация_сердца'].values()), key='auscultation_heart')
        auscultation_heart_key = [k for k, v in demographic_keys['Аускультация_сердца'].items() if v == auscultation_heart_label][0]
    with col_resp_rate:
        resp_status_label = st.selectbox('Дыхание (Частота/Характер)', list(demographic_keys['Дыхание_паттерн'].values()), key='resp_status')
        resp_status_key = [k for k, v in demographic_keys['Дыхание_паттерн'].items() if v == resp_status_label][0]

    col_lung_ausc, col_resp_type, col_trachea_refl, col_tono = st.columns(4)
    with col_lung_ausc:
        auscultation_lung_label = st.selectbox('Аускультация легких', list(demographic_keys['Аускультация_легких'].values()), key='auscultation_lung')
        auscultation_lung_key = [k for k, v in demographic_keys['Аускультация_легких'].items() if v == auscultation_lung_label][0]
    with col_resp_type:
        resp_type_label = st.selectbox('Тип дыхания', list(demographic_keys['Тип_дыхания'].values()), key='resp_type')
        resp_type_key = [k for k, v in demographic_keys['Тип_дыхания'].items() if v == resp_type_label][0]
    with col_trachea_refl:
        trachea_reflex = st.radio('Трахеальный рефлекс', ['Да', 'Нет'], key='trachea_reflex')
    with col_tono:
        tonometry_status_label = st.selectbox('Тонометрия', list(demographic_keys['Тонометрия'].values()), key='tonometry_status')
        tonometry_status_key = [k for k, v in demographic_keys['Тонометрия'].items() if v == tonometry_status_label][0]


# --- 3.3. КЛИНИЧЕСКИЕ СИМПТОМЫ (ЖАЛОБЫ) ---
with tab_symptoms:
    st.header('Клинические симптомы (Жалобы владельца)')
    st.markdown('**Выберите все наблюдаемые симптомы в соответствующих вкладках.**')

    selected_symptoms = {}
    
    symptom_tab_list = st.tabs(list(symptom_groups.keys()))

    for tab, (group_name, symptoms_list) in zip(symptom_tab_list, symptom_groups.items()):
        with tab:
            st.markdown(f"**{group_name}**")
            num_cols = 4
            cols = st.columns(num_cols)
            
            for i, (symptom_key, display_label) in enumerate(symptoms_list):
                with cols[i % num_cols]:
                    selected_symptoms[symptom_key] = st.checkbox(display_label, key=f"symptom_{symptom_key}")


st.markdown('---')

# --- 4. ЛОГИКА ПРЕДСКАЗАНИЯ (Сбор всех детализированных ключей) ---
if st.button('РАССЧИТАТЬ ПРЕДВАРИТЕЛЬНЫЙ ДИАГНОЗ', type="primary", use_container_width=True):
    
    with st.spinner('Сбор детализированного протокола...'):
        
        # --- 4.1. Сбор ВСЕХ деталей для новой модели (150+ признаков) ---
        final_input_dict = {}
        
        # 1. Демография (OHE ключи)
        final_input_dict[f'Порода_{breed_key}'] = 1
        final_input_dict[f'Возраст_{age_key}'] = 1
        final_input_dict[f'Пол_{gender_key}'] = 1
        final_input_dict['Кастрация_Да'] = 1 if is_castrated else 0
        
        # 2. Анамнез (OHE + Бинарные)
        final_input_dict['Длительность_владения_Более_года'] = 1 if ownership_duration == 'Более года' else 0 
        final_input_dict[f'Содержание_{housing_label.split(" ")[0].lower()}'] = 1 
        final_input_dict[f'Диета_{diet_label.split(" ")[0].lower()}'] = 1 
        
        final_input_dict['Хронические_заболевания_Да'] = 1 if has_chronic else 0
        final_input_dict['Перенесенные_операции_заболевания_Да'] = 1 if has_surgery else 0
        final_input_dict['Антибиотикоанамнез_Да'] = 1 if has_antibio else 0
        final_input_dict['Аллергия_Да'] = 1 if allergy_status else 0
        
        final_input_dict['Дегельминтизация_Да'] = 1 if deworm_status == 'Да' else 0
        final_input_dict['Обработка_от_блох_клещей_Да'] = 1 if ectopara_status == 'Да' else 0
        final_input_dict['Сожители_с_симптомами_Да'] = 1 if cohabiting_status == 'Да, с симптомами' else 0

        # Вакцинация (OHE ключи)
        final_input_dict[f'Вирусная_вакцина_{vaccine_selection_virus_key}'] = 1
        final_input_dict[f'Бешенство_вакцина_{vaccine_selection_rabies_key}'] = 1


        # 3. Vitals (OHE ключи)
        final_input_dict[f'Общее_состояние_{overall_status_key}'] = 1
        final_input_dict[f'Ментальный_статус_{mental_status_key}'] = 1
        final_input_dict[f'Вес_кондиция_{weight_key_vitals}'] = 1
        final_input_dict[f'Температура_тела_{temp_key_vitals}'] = 1
        final_input_dict[f'Пульс_{pulse_status_key}'] = 1
        final_input_dict[f'Аускультация_сердца_{auscultation_heart_key}'] = 1
        final_input_dict[f'Дыхание_паттерн_{resp_status_key}'] = 1
        final_input_dict[f'Аускультация_легких_{auscultation_lung_key}'] = 1
        final_input_dict[f'Тип_дыхания_{resp_type_key}'] = 1
        final_input_dict[f'Дегидратация_{dehydration_status_key}'] = 1
        final_input_dict[f'КНС_{kns_status_key}'] = 1
        final_input_dict[f'Тонометрия_{tonometry_status_key}'] = 1
        final_input_dict['Трахеальный_рефлекс_Да'] = 1 if trachea_reflex == 'Да' else 0


        # 4. Клинические симптомы (Бинарные ключи)
        for symptom_key, is_selected in selected_symptoms.items():
            final_input_dict[symptom_key] = 1 if is_selected else 0


        # --- 4.2. Преобразование и предсказание (РЕАЛЬНАЯ ЛОГИКА 15) ---
        
        # 1. Создаем пустой DataFrame со всеми признаками в нужном порядке
        input_df_aligned = pd.DataFrame(0, index=[0], columns=TRAINING_FEATURES)
        
        # 2. Заполняем DataFrame выбранными пользователем значениями (1)
        for feature, value in final_input_dict.items():
            if feature in TRAINING_FEATURES:
                input_df_aligned.loc[0, feature] = value
        
        # 3. Применяем препроцессор (StandardScaler)
        processed_input = preprocessor.transform(input_df_aligned)
        
        # 4. Предсказание
        predictions_proba = model.predict(processed_input)
        predictions = predictions_proba[0]
        
        # Берем ТОП-2 диагноза
        top_2_indices = predictions.argsort()[-2:][::-1]
        top_2_probabilities = predictions[top_2_indices]
        top_2_diagnoses = label_encoder.inverse_transform(top_2_indices)

    st.markdown('## 4. Результат Анализа (Протокол)')
    
    st.markdown('### 💡 Предварительные диагнозы')
    
    # Функция для форматирования диагноза: Русский + (Латынь/Ключ)
    def format_diagnosis(key):
        return DIAGNOSIS_TRANSLATIONS.get(key, key.replace('_', ' ').capitalize() + f" (Key: {key})")
    
    # Первый диагноз
    diagnosis_1_key = top_2_diagnoses[0]
    diagnosis_1_display = format_diagnosis(diagnosis_1_key)
    prob_1 = top_2_probabilities[0]
    st.success(f"**Предварительный Диагноз 1:** {diagnosis_1_display} — **{prob_1*100:.1f}%**")
    
    # Второй диагноз
    if len(top_2_diagnoses) > 1:
        diagnosis_2_key = top_2_diagnoses[1]
        diagnosis_2_display = format_diagnosis(diagnosis_2_key)
        prob_2 = top_2_probabilities[1]
        st.info(f"**Предварительный Диагноз 2 (Дифференциальный):** {diagnosis_2_display} — {prob_2*100:.1f}%")

        # --- ИНТЕГРАЦИЯ SHAP В APP.PY ---
    st.markdown("### 🔍 Интерпретация решения")
    with st.expander("Анализ влияния симптомов на диагноз"):
        try:
            import shap
            import matplotlib.pyplot as plt
            from sklearn.linear_model import LinearRegression

                    # 1. Функция-предиктор
            def map_predict(x):
            # verbose=0 убирает лишний лог в консоль сервера
                preds = model.predict(x, verbose=0)
                return preds[:, top_2_indices[0]]

                    # 2. Фоновое состояние
            background = np.zeros((1, len(TRAINING_FEATURES)))
                    
                    # 3. Инициализируем Explainer
            explainer = shap.KernelExplainer(map_predict, background)
                    
                    # 4. РАСЧЕТ С ИСПРАВЛЕНИЕМ ОШИБКИ: 
                    # Мы указываем l1_reg="num_features(10)", чтобы SHAP выбрал 10 главных признаков 
                    # и не пытался строить сложную Lasso-модель на 150 признаках при 1 семпле.
            shap_vals = explainer.shap_values(
                processed_input, 
                nsamples=100, 
                l1_reg="num_features(10)"
            )

                    # 5. Визуализация
            col_chart_1, col_chart_2, col_chart_3 = st.columns([1, 2, 1])
            with col_chart_2:
                fig, ax = plt.subplots(figsize=(6, 4))
                    
                    # Отрисовка
                shap.summary_plot(
                    shap_vals, 
                    processed_input, 
                    feature_names=TRAINING_FEATURES, 
                    plot_type="bar", 
                    max_display=8, 
                    show=False
                )
                    
                plt.title(f"Вклад признаков в диагноз: {diagnosis_1_key}")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            st.caption("График показывает 8 наиболее значимых симптомов для данного случая.")
                
        except Exception as e:
            st.error(f"Ошибка SHAP: {e}")

    st.markdown('---')
    
    # --- ВЫВОД ПРОТОКОЛА ---
    
    # Логика сбора вакцин для вывода
    final_vax_output = []
    if vax_status == 'Да':
        if vaccine_selection_virus_key != 'Нет_вирусной_вакцинации':
            final_vax_output.append(f"Вирусная/Комплексная: {vax_virus_label}")
            
            # ИСПРАВЛЕНИЕ: Используем ранее определенные переменные is_complex_rabies_dog/cat
            if is_complex_rabies_dog or is_complex_rabies_cat:
                final_vax_output.append(f"Бешенство: {vax_virus_label} (Комплексная)")
            elif vaccine_selection_rabies_key != 'Нет_бешенства':
                 vax_rabies_label = [v for k, v in VACCINES_DOGS_RABIES.items() if k == vaccine_selection_rabies_key][0] if breed_key == 'собака' else [v for k, v in VACCINES_CATS_RABIES.items() if k == vaccine_selection_rabies_key][0]
                 final_vax_output.append(f"Бешенство: {vax_rabies_label}")

    final_vax_display = '; '.join(final_vax_output) if final_vax_output else 'Вакцинация не проведена'
    
    # Сбор выбранных симптомов для вывода в Протокол
    active_symptoms_list = [f"✅ {label}" for key, group in symptom_groups.items() for s_key, label in group if selected_symptoms.get(s_key)]
    
    st.markdown('### 📝 Протокол обследования')
    
    st.markdown("#### Анамнез и История")
    col_an_1, col_an_2, col_an_3, col_an_4 = st.columns(4)
    col_an_1.markdown(f"**Вид/Возраст:** {breed_label} / {age_label_selected}")
    col_an_2.markdown(f"**Пол:** {gender_label_selected} ({'Кастрирован/на' if is_castrated else 'Не кастрирован/на'})")
    col_an_3.markdown(f"**Содержание/Диета:** {housing_label} / {diet_label}")
    col_an_4.markdown(f"**Хронич./Аллергия:** {'Да' if has_chronic else 'Нет'} / {'Да' if allergy_status else 'Нет'}")
    st.markdown(f"**Профилактика:** Дегельм: {deworm_status}; Экто: {ectopara_status}; Сожители: {cohabiting_status}")
    st.markdown(f"**Вакцинация:** {final_vax_display}")
    
    st.markdown("#### Клиническое обследование")
    col_v_1, col_v_2, col_v_3 = st.columns(3)
    col_v_1.markdown(f"**Общее состояние:** {overall_status_label} ({mental_status_label})")
    col_v_2.markdown(f"**Вес/Кондиция:** {weight_label}")
    col_v_3.markdown(f"**Температура:** {temp_label}")
    
    col_v_4, col_v_5, col_v_6 = st.columns(3)
    col_v_4.markdown(f"**Пульс:** {pulse_status_key} / **Сердце (Ауск.):** {auscultation_heart_key}")
    col_v_5.markdown(f"**Дыхание (Частота/Тип):** {resp_status_key} / {resp_type_key}")
    col_v_6.markdown(f"**Легкие (Ауск.):** {auscultation_lung_key} / **Трах. Реф.:** {trachea_reflex}")
    
    col_v_7, col_v_8, col_v_9 = st.columns(3)
    col_v_7.markdown(f"**Дегидратация:** {dehydration_status_key}")
    col_v_8.markdown(f"**КНС:** {kns_status_key}")
    col_v_9.markdown(f"**Тонометрия:** {tonometry_status_key}")

    st.markdown("#### Клинические симптомы (Выбранные)")
    
    if active_symptoms_list:
        cols_symp = st.columns(4)
        for i, symp in enumerate(active_symptoms_list):
            cols_symp[i % 4].markdown(symp)
    else:
        st.markdown("*Симптомы не выбраны.*")
    
    # Подготовка данных для PDF
    report_payload = {
        'diag1': diagnosis_1_display,
        'prob1': f"{prob_1*100:.1f}%",
        'diag2': diagnosis_2_display if len(top_2_diagnoses) > 1 else "Нет",
        'breed': breed_label,
        'age': age_label_selected,
        'vax': final_vax_display,
        'preventive': f"Дегельм: {deworm_status}; Экто: {ectopara_status}",
        'status': f"{overall_status_label} ({mental_status_label})",
        'weight': weight_label,
        'temp': temp_label,
        'heart': auscultation_heart_key,
        'pulse': pulse_status_key,
        'resp': resp_status_key,
        'symptoms': [s.replace("✅ ", "") for s in active_symptoms_list]
    }

    # Кнопка скачивания
    pdf_bytes = export_to_pdf(report_payload)
    st.download_button(
        label="📄 Скачать PDF протокол",
        data=pdf_bytes,
        file_name=f"VetAI_{breed_label}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    st.markdown("---")
    st.caption(f"**Версия:** v0.4.1 (Alpha). **Работает на модели v{MODEL_VERSION}.** Требуется полный клинический и лабораторный анализ для постановки окончательного диагноза.")
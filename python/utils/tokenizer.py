# создадим словарь аббревиатур
ABBREVIATIONS = {'мсфо':'международные стандарты финансовой отчетности',
        'рсбу':'российские стандарты бухгалтерского учета',
        'гвк':'голос внутреннего клиента',
        'ипт':'информационно платежный терминал',
        'зро':'заместитель руководителя офиса',
        'ро':'руководитель офиса',
        'осб':'отделение сбербанка',
        'спп':'специалист прямых продаж',
        'рко':'расчетно кассовое обслуживание',
        'пко':'приходный кассовый ордер',
        'зод':'завершение операционного дня',
        'зсо':'зона самообслуживания',
        'прпа':'подразделения по работе с проблемными активами',
        'прпз':'подразделения по работе с проблемной задолженностью',
        'ммб':'международный московский банк',
        'цундо':'центр управления наличным денежным обращением',
        'цкп':'центр комплексной поддержки',
        'цско':'центр сопровождения клиентских операций',
        'мфц':'многофункциональный центр',
        'пф':'пенсионный фонд',
        'фот':'фонд оплаты труда',
        'цчб':'центрально черноземный банк',
        'хцк':'хранилище ценностей клиентов',
        'эп':'электронная подпись',
        'цп':'цифровая подпись',
        'эцп':'электронно цифровая подпись',
        'зно':'заявка на обслуживание',
        'лл':'lean лаборатория',
        'епд':'единый платежный документ',
        'есзк':'единый сервис заботы о клиентах',
        'псо':'потенциально сомнительные операции',
        'умпо':'управление мониторинга подозрительных операций',
        'рцб':'рынок ценных бумаг',
        'прко':'проект коллегиального органа банка',
        'битл':'банк инсайдер третьих лиц',
        'кнн':'коэффициент налоговой нагрузки',
        'оппп':'отдел поддержки прямых продаж',
        'пцп':'подразделение центрального подчинения',
        'пше':'производственная штатная единица',
        'суо':'система управления очередью',
        'тб':'территориальный банк',
        'ца':'центральный аппарат',
        'ипр':'индивидуальный план развития',
        'жкх':'жилищное коммунальное хозяйство',
        'инн':'идентификационный номер налогоплательщика',
        'огрн':'основной государственный регистрационный номер',
        'нкд':'накопленный купонный доход',
        'тс':'торговая система',
        'псс':'производственная система сбербанка',
        'црт':'центр речевых технологий',
        'оад':'оперативный архив документов',
        'бфо':'безбумажный фронт офис',
        'уп':'условный продукт',
        'цопп':'центр оперативной поддержки продаж',
        'ап':'автоматический платёж',
        'дс':'денежные средства',
        'тмц':'товарно материальные ценности',
        'спообк':'система предварительной обработки операций банковских карт',
        'аспк':'автоматизированная система повышения квалификации',
        'смо':'старший менеджер по обслуживанию',
        'кбп':'консультант по банковским продуктам',
        'мпп':'менеджер по продажам',
        'рг':'руководитель группы',
        'рп':'руководитель проектов',
        'фс':'фронтальная система',
        'ефс':'единая фронтальная система',
        'пк':'персональный компьютер',
        'киц':'кассово инкассаторский центр',
        'дрпа':'департамент работы с проблемными активами',
        'дгр':'департамент глобальных рынков',
        'дмик':'департамент маркетинга и коммуникаций',
        'дрок':'департамент развития отношений с клиентами',
        'осб':'отделение сбербанка',
        'осц':'объединенный сервисный центр',
        'окб':'объединенное кредитное бюро',
        'фл':'физические лица',
        'фнс':'федеральная налоговая служба',
        'црб':'центр развития бизнеса',
        'тз':'техническое задание',
        'кхд':'корпоративное хранилище данных',
        'ндс':'налог на добавленную стоимость',
        'ндфл':'налог на доходы физических лиц',
        'мимс':'малой и мобильной сети',
        'ргмимс':'руководитель группы малой и мобильной сети',
        'мач':'миллиампер час',
        'убуио':'управление бухгалтерского учета и отчетности',
        'екп':'единая карта петербуржца',
        'урм':'удаленное рабочее место',
        'сбсж':'сбербанк страхование жизни',
        'сбт':'сбербанк технологии',
        'ус':'устройство самообслуживания',
        'ису':'интеллектуальная система управления',
        'ас':'автоматизированная система',
        'фио':'фамилия имя отчество',
        'ио':'исполняющий обязанности',
        'лого':'логотип',
        'доп':'дополнительный',
        'перс':'персональный',
        'гос':'государственный',
        'корп':'корпоративный',
        'юр':'юридический',
        'физ':'физический',
        'арх':'архитектурный',
        'пром':'промышленный',
        'др':'другой',
        'эл':'электронный',
        'мед':'медицинский',
        'ден':'денежный',
        'мес':'месяц',
        'тер':'территориальный',
        'соц':'социальный',
        'орг':'организационный',
        'зам':'заместитель',
        'юл':'юридическое лицо',
        'фл':'физическое лицо',
        'кк':'кредитная карта',
        'ии':'искусственный интеллект',
        'кц':'call центр',
        'рб':'розничный бизнес',
        'кб':'корпоративный бизнес',
        'киб':'корпоративный и инвестиционный бизнес',
        'бд':'база данных',
        'егрн':'единый государственный реестр недвижимости',
        'егрп':'единый государственный реестр прав',
        'егрип':'единый государственный реестр индивидуальных предпринимателей',
        'егрюл':'единый государственный реестр юридических лиц',
        'гибдд':'государственная инспекция безопасности дорожного движения',
        'гаи':'государственная авто инспекция',
        'егпо':'единое гибридное программное обеспечение',
        'есиа':'единая система идентификации и аутентификации',
        'пнр':'плановый номер релиза',
        'фссп':'федеральная служба служебных приставов',
        'па':'проблемные активы',
        'ермб':'единый розничный мобильный банк',
        'бки':'бюро кредитных историй',
        'смэв':'система межведомственного электронного взаимодействия',
        'всп':'внутреннее структурное подразделение',
        'рвсп':'руководитель внутреннего структурного подразделения',
        'зрвсп':'заместитель руководителя внутреннего структурного подразделения',
        'дбо':'дистанционное банковское обслуживание',
        'екс':'единая корпоративная система',
        'цуп':'центр управления проектами',
        'лк':'личный кабинет',
        'лкк':'личный кабинет клиента',
        'ук':'управляющий комитет',
        'ду':'доверительное управление',
        'бк':'банковская карта',
        'епс':'единая платежная система',
        'бмо':'блок международные операции',
        'уко':'удаленные каналы обслуживания',
        'иис':'индивидуальный инвестиционный счет',
        'нсж':'накопительное страхование жизни',
        'исж':'инвестиционное страхование жизни',
        'ипп':'индивидуальный пенсионный план',
        'дзо':'дочернее зависимое общество',
        'нпф':'негосударственный пенсионный фонд',
        'опс':'обязательное пенсионное страхование',
        'пиф':'паевой инвестиционный фонд',
        'ип':'индивидуальный предприниматель',
        'ооо':'общество с ограниченной ответственностью',
        'кэип':'кредитный эксперт по ипотечному кредитованию',
        'ериб':'единый розничный интернет банк',
        'цнс':'центр недвижимости сбербанка',
        'цзк':'центр заботы о клиентах',
        'лкз':'личный кабинет заемщика',
        'пбд':'первичная бухгалтерская документация',
        'ппрб':'платформа поддержки развития бизнеса',
        'оэср':'организация экономического сотрудничества и развития',
        'еасуп':'единая автоматизированная система управления персоналом ',
        'еркц':'единый распределенный контактный центр',
        'пси':'приемо сдаточные испытания',
        'сбт':'сбербанк технологии',
        'пмз':'подразделение мониторинга залогов',
        'ндфл':'налог на доходы физических лиц',
        'аис':'автоматическая идентификационная система',
        'асв':'агенство по страхованию вкладов',
        'гис':'географическая информационная платформа',
        'эдо':'электронный документооборот',
        'фп':'функциональная подсистема',
        'ксш':'коропративная сервисная шина',
        'скб':'северо кавказский банк',
        'ксб':'крупный и средний бизнес',
        'дбт':'детальные бизнес требования',
        'цуп':'центр управления проектами',
        'пнр':'плановый номер релиза',
        'осс':'оператор сотовой связи',
        'пбт':'подробные бизнес требования',
        'дсж':'добровольное страхование жизни',
        'мрм':'мобильное рабочее место',
        'уко':'удаленные каналы обслуживания',
        'пфр':'пенсионный фонд россии',
        'кпэ':'ключевой показатель эффективности',
        'kpi':'ключевой показатель эффективности',
        'есзк':'единая служба заботы о клиентах',
        'ппр':'приоритетные проекты руководителя',
        'зни':'запрос на изменение',
        'мик':'менеджер по ипотечному кредитованию',
        'абс':'автоматизированная банковская система',
        'умм':'учебно методические материалы',
        'сбс':'сбербанк сервис',
        'вкс':'видео конференцсвязь',
        'ткс':'теле конференцсвязь',
        'сб':'сбербанк',
        'гб':'гигабайт',
        'внд':'внутренний нормативный документ',
        'ссср':'советский союз',
        'рф':'россия',
        'спб':'санкт петербург',
        'мвд':'министерство внутренних дел',
        'фз':'федеральный закон',

        'ui':'user interface',
        'ux':'user experience',
        'nps':'net promoter score',
        'csi':'customer satisfaction index',
        'erm':'enterprise risk management',
        # 'ip':'internet protocol',
        # 'url':'uniform resourse locator',
        # 'api':'application programmimg interface',
        'cf':'cash flow',
        'fatca':'foreign account tax complience act',
        'sbi':'sberbank international',
        'mdes':'mastercard digital enablement service',
        'ai':'artificial intelligence', # искусственный интеллект (при вводе ai и ии будут 2 разных результата)
        'atm':'automatic teller machine',
        'sla':'service level agreement',
        'crm':'customer relationship management',
        'mvp':'minimal viable product',
        # 'devops':'development operations',
        # 'девопс':'development operations',
        'crs':'common reporting standart ',
        'pfm':'personal finance management',
        'ivr':'interactive voice response',
}

# импортируем необходимые библиотеки
import re
import joblib
from itertools import compress
# import pymorphy2
from pymystem3 import Mystem
import app_env
# import ast

morph = Mystem()

re_html = r'<.*?>'
re_site = r'https?://\S+|www\.\S+'
re_mail = r'[\w\.-]+@[\w\.-]+\.\w+'
re_hex = r'[\x80-\xff]+'


def remove_html(text):
    html_pattern = re.compile(re_html)
    return html_pattern.sub(' ', text)
     
def remove_hex(text):
    hex_pattern = re.compile(re_hex)
    return hex_pattern.sub(' ', text)

def remove_site(text):
    site_pattern = re.compile(re_site)
    return site_pattern.sub(' ', text)

def remove_mail(text):
    mail_pattern = re.compile(re_mail)
    return mail_pattern.sub(' ', text)

def re_pattern(st):
    return '^{0}\W+|^{0}$|\W+{0}$|\W+{0}\W+'.format(st)


stopwords = joblib.load(app_env.data_model_runtime_path('stopwords'))
stopwords =  stopwords +\
['gt', 'nan','др', 'го', 'мб', 'пр', 'тд', 'тк',
'ваш', 'её','весь', 'весьма','вовсе', 'дабы', 'например', 'ибо', 'имхо',
'зато', 'ещё', 'зря', 'десятка','сотня', 'третье', 'тысяча', 'который',
'кстати', 'либо', 'ль', 'наш', 'нем', 'оно', 'особо', 'очень', 'поэтому', 'причём',
'свой', 'скажем', 'слишком','столь', 'сюда', 'также', 'твой', 'точнее', 'хотя', 'чей', 'это', 'якобы']
#надо в sw, нужно - нет
stopwords = list(set(stopwords))
stopwords = [a for a in stopwords if a!='много']


# REPLACE DICTS

replace_dict_re_upcase =  {
    
        re_pattern('ПО'):' програмное обеспечение ',
        re_pattern('КМ'):' клиентский менеджер ',
        re_pattern('СМ'):' сервис менеджер ',
        re_pattern('ДУЛ'):' документ удостоверяющий личность ',
        re_pattern('ДО'):' дополнительный офис ',    
        re_pattern('МБ'):' мобильный банк ',
        re_pattern('ВШ'):' виртуальная школа ',
        re_pattern('КУ'):' корпоративный университет ',
        re_pattern('ИСК'):' информационные сведения клиента ',
        re_pattern('ТО'):' техническое обслуживание ',
        re_pattern('ПИР'):' плановый интеграционный релиз ',
        re_pattern('ПК ТБ'):' подразделение compliance территориального банка ',
     
        '\W+СОП[а-я]{0,3}|^СОП[а-я]{0,3}': ' стандартные операционные процедуры ',
        '^ГОСБ[а-я]{0,3}|\W+ГОСБ[а-я]{0,3}': ' головное отделение сбербанка ', 
        '^АИБ[а-я]{0,3}|\W+АИБ[а-я]{0,3}': ' администратор информационной безопасности ', 
    
        'ДУС[а-я]{1}': 'доброе утро сбербанк', 
            
        'КУРС[а-я]{0,2}': 'мобильное приложение сбербанка курс', 
        'ДРУГ[а-я]{0,2}': 'дирекция распределенных услуг друг',
    
        'ПОД/ФТ': 'противодействие легализации доходов полученных преступным путем и финансированию терроризма', 
        'M&A': 'сделки по слияниям и поглощениям', 
        'КОРТ': 'квартальный обзор результатов трайба', 
        
}

replace_dict_re_lowcase = {
    
        '"сбербанк"': 'сбербанк',
        'пао сбербанк': 'сбербанк',
    
        'онл@йн': 'онлайн',
        'wi-fi':' wifi ',
    
        'колцентр':'call-центр',
    
        'тк рф': ' трудовой кодекс россии',
        'гк рф': ' гражданский кодекс россии',

        'сим[ -]карт': "sim карт",
        '\d+ мин[ .]': 'минут ', 

        'орг[ .]{1,3}тех': ' офисная тех',
        'зар[ .]{1,3}плат': ' заработная плат',
                
        'кол-в\w{0,3}':' количество ',
        'ср-в\w{0,3}':' средство ',
        'рук-л\w{1,3}':' руководитель ',    
        'соцсет\w{1,3}': ' социальная сеть ',
        
        'сбол\w{0,3}': ' сбербанк онлайн ',
        'сббол\w{0,3}': ' сбербанк бизнеc онлайн ',
        'онлайн\w{0,3}': ' онлайн ',
        'лайфхак\w{0,3}': ' лайфхак ',
    
        'зарплат[^н]{1,3}': ' заработная плата ', # зарплатный
        '\W+зарплат\W+': ' заработная плата ', 
    
        'рук-л\w{1,2}': ' руководитель ',
    
        'тербанк\w{0,3}': ' территориальный банк ',
    
        '^орг-и\w{0,3}|\W+орг-и\w{0,3}|^орг-ц\w{1,3}|\W+орг-ц\w{1,3}': ' организация ', 
    
        re_pattern('сб@'): ' сбрербанк онлайн ',
        re_pattern('зп'): ' заработная плата ',

        re_pattern("b2b"):" business to business ",
        re_pattern("b2c"):" business to consumer ",
        re_pattern("b2g"):" business to government ",
    
        re_pattern('5+'): " система оценки личной эффективности ",
        re_pattern("c+\+"):" компилируемый язык программирования ",
        re_pattern("топ-200"):" ключевые менеджеры банка ",
    
        re_pattern('срм'): ' crm ',
        re_pattern('сrm'): ' crm ',
    
#         'фронт[ -]офис\w{0,2}': 'front офис',
#         'бэк[ -]офис\w{0,2}': 'back офис',
#         кэшбек, кэш бек
   
}

stop_phr_list = [

        'ссылка на идею',
        # 'в наше отделение обратился клиент',
        #  'клиент остался \w+ доволен', 'клиент остался доволен',
        'добрый день', 'хочу поделиться историей', 'уважаемые коллеги', 'обычный рабочий день',
        'на самом деле', 'может быть', 'таким образом', 'скорее всего',
         'проще говоря', 'честно говоря', 'по правде говоря', 'грубо говоря', 'другими словами', 'в свою очередь', 'в принципе',
        'во первых', 'во вторых', 'в третьих',
        'как выяснилось', 'как оказалось', 'как следствие',
        'на \w+ взгляд',
        'по крайней мере', 'помимо этого', 'прежде всего', 'с другой стороны', 'кроме того',
#         ввиду того что, в связи с тем что
]

replace_lem_dict = {
#             'банка': 'банк',
            'клиентка': 'клиент',
            'смочь': 'мочь',
            'сбер': 'сбербанк',
            'многие':'многий',
            'совершать':'совершить',
#             'клавиш':'клавиша',
            'ms':'microsoft',
#             'авить':'авито',
#             'минь':'мини',
#             'поль':'поле',
#             'плат':'плата',
#             'вип':'vip',
#             'чаять':'чай',
#             'банкнот':'банкнота',
            'online':'онлайн',
            'моб':'мобильный',
            'сап':'sap',
#             'стен':'стена',
            'фрод':'fraud',
            'ит':'it',
            'страхов':'страховой',
            'смс':'sms',
#             'вод':'вода',
#             'мор':'море',
#             'скора':'скорый',
#             'пришлый': 'прийти',
            'ok':'ок',
            'лин':'lean',
            'dev':'development',
            'фреймворк':'framework',
            'эджайл':'agile',
            'скрам':'scrum',
            'комплайнс':'compliance',
            'девопс':'devops',
            'руб': 'рубль',
    
}

def tokenize(st):
    
    st = str(st)
    
    st = st.encode('utf-8').replace(b'\xcc\x81', b'').decode('utf-8') # удаление ударений
    
    for kk, vv in replace_dict_re_upcase.items():
        st = re.sub(kk, vv, st)
        
#     st = " ".join([replace_dict[a] if a in replace_dict_split_upcase.keys() else a for a in re.split('[^а-яА-Яa-zA-ZЁё]', st)])
    #     неоптимальный пербор лишь для 3 абр-р
        
    st = st.lower()
    
    st = remove_html(st)
    st = remove_site(st)
    st = remove_mail(st)
    st = remove_hex(st)
    
    for stop_phr in stop_phr_list:
        st = re.sub(stop_phr, '', st)
        
    for kk, vv in replace_dict_re_lowcase.items():
        st = re.sub(kk, vv, st)

#     раскрытие слов со спецсимволами
#     st = " ".join([replace_dict_split[a] if a in replace_dict_split.keys() else a for a in re.split('[ \t\n.,;:()]', st)])
    
    st = re.sub('[^a-zа-яё]+', ' ', st) # '[^A-Za-zА-Яа-яёЁ]+'
    st = " ".join([ABBREVIATIONS[a] if a in ABBREVIATIONS.keys() else a for a in st.split()])
    st = re.sub('[^a-zа-яё]+', ' ', st) # '[^A-Za-zА-Яа-яёЁ]+'
    
    st_list = st.split()
    st_list = [w for w in st_list if w not in stopwords and len(w)>1]

    st_list_final = []
    lemmas = morph.analyze(" ".join(st_list))
    for lemma in lemmas:
        analysis = lemma.get('analysis')
        if analysis:
            for lex in analysis:
                gr = re.split('\W+', lex.get('gr'))[0]
                lexx = lex.get('lex')
                if gr not in ['NUM', 'SPRO', 'PR', 'PART', 'INTJ', 'ANUM'] or lexx in ['спасибо', 'мало', "много"]:
    #                 'CONJ'
                    st_list_final.append(lexx)   
        else:
            st_list_final.append(lemma.get('text'))  
            

    st_list_final = [w for w in st_list_final if w not in stopwords and len(w)>1]

    st_list_final = [replace_lem_dict.get(item, item) for item in st_list_final]
    
    return st_list_final

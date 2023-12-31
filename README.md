## Добро пожаловать в RL_Lib: библиотеку обучения с подкреплением на базе TensorFlow!

RL_Lib - это мощный и гибкий инструмент для обучения алгоритмов с подкреплением с использованием моделей на TensorFlow.\
Мы предоставляем реализацию нескольких популярных алгоритмов обучения с подкреплением, которые легко можно интегрировать в собственные проекты или использовать в нашей готовой обертке обучения.



### Основные возможности RL_Lib:
<ol>
    <li>Реализация различных алгоритмов: RL_Lib предоставляет готовые реализации DQN, DRQN, QR-DQN и Ape-X. Вы можете использовать их "из коробки" или настраивать под свои нужды.</li>

<li>Интеграция с TensorFlow: Вам не нужно изучать новый фреймворк. Все алгоритмы написаны с использованием TensorFlow, поэтому вы можете использовать свои собственные модели на базе Keras, TensorFlow или любые другие совместимые модели.</li>

<li>Гибкость и настраиваемость: RL_Lib предоставляет множество параметров и опций для настройки обучения алгоритмов под ваши задачи.</li>

<li>Мощные буферы: Для хранения и обработки данных RL_Lib предоставляет различные буферы, включая обычные и приоритетные буферы, а также n-step буферы.</li> 

<li>Сохранение этапов обучения: Вы можете сохранять прогресс обучения для последующего использования или воспроизведения.</li>

<li>Использование обертки обучения: Упростите процесс обучения, используя готовую обертку, которая выполняет все необходимые действия в среде для обучения алгоритма. Она автоматически обрабатывает состояния, действия, награды и т. д.</li>

</ol>

Мы стремимся предоставить простой, но мощный инструмент для обучения агентов с подкреплением, который позволит вам быстро и эффективно исследовать различные алгоритмы и применять их в реальных задачах. Добро пожаловать в RL_Lib - ваш надежный партнер в обучении с подкреплением!

## Текущие алгоритмы
<ul>
    <li>DQN и его модификации</li>
    <li>DRQN</li>
    <li>DDPG</li>
</ul>

## Базовое использование
#### Создание алгоритма по умолчанию (конфиг можно посмотреть в папке алгоритма):
```
from rl_lib.src.algoritms.dqn.dqn import DQN

config = {'model_config':{}}
config['model_config']['input_shape'] = env.observation_space.shape
config['model_config']['action_space'] = env.action_space.n

algo = DQN(config)
```

#### Создание алгоритма пользовательского алгоритма:
```
from rl_lib.src.algoritms.dqn.dqn import DQN
from yaml import safe_load

path = #путь к файлу конфигурации

config = safe_load(
            open(
                os_path.join(
                        os_path.dirname(path),"./config.yaml"
                            ),
                "rb")
                )
config['model_config']['input_shape'] = env.observation_space.shape
config['model_config']['action_space'] = env.action_space.n

algo = DQN(config)
```
## Основные методы алгоритма
#### Сохранение и загрузка сохраненного алгоритма:
```
algo.save()
algo.load()
```
#### Предсказание действия с политикой исследования:
```
algo.get_action(obs)
```


#### Предсказание тестового действия:
```
algo.get_test_action(obs)
```

#### Обучение:
```
sardsn = (obs, action, reward, done, next_obsv) #именно в таком порядке
algo.add(sardsn)

algo.train_step() #Один шаг грандиентного спуска
algo.copy_weights() #Копирование весов learner -> target (если tau == 1.0)
```

#### Обновление внутреннего состояния алгоритма происходит автоматически. \
Для рекуррентной модели drqn метод инициализации состояния:
```
algo.initial_state()
```

## Пример использования можно посмотреть:
```
└──examples
    ├──dqn
        └──cart_pole
            ├──config.yaml
            └──dqn_cart_pole.py
    └──drqn
        └──cart_pole
            ├──config.yaml
            └──drqn_cart_pole.py
```

## Будущее проекта
<ul type="disk">
    <li>Реализация алгоритмов:
    <ul>
        <li>QR-DQN</li>
        <li>IQN</li>
        <li>A2C</li>
        <li>TD3</li>
        <li>Ape-X</li>
        <li>RD2D</li>
        <li>Bandits</li>
    </ul>
    <li>Добавление LaziFrames в буферы сохранения</li>
    <li>Написание обертки шагов обучения в среде</li>
    <li>Реализация записи статистики обучения</li>
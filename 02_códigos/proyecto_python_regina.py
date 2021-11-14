{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Limpieza de datos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cargar paquetería necesaria \n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "#import seaborn as sns\n",
    "import preprocessor as p\n",
    "\n",
    "#from openpyxl import Workbook\n",
    "#import matplotlib.pyplot as plt\n",
    "#import microtc as microtc\n",
    "#from microtc.textmodel import TextModel\n",
    "from sklearn.metrics import f1_score, accuracy_score\n",
    "from sklearn.svm import LinearSVC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importar datos\n",
    "tuits_1 = pd.read_json(\"tuits001.txt\", lines = True) # @Mary_Luisa_AG  \n",
    "tuits_2 = pd.read_json(\"tuits002.txt\", lines = True) # @LuRiojas\n",
    "tuits_3 = pd.read_json(\"tuits003.txt\", lines = True) # @vmva1950\n",
    "tuits_4 = pd.read_json(\"tuits004.txt\", lines = True) # @aostoaortega\n",
    "tuits_5 = pd.read_json(\"tuits005.txt\", lines = True) # @mariamercedg\n",
    "tuits_6 = pd.read_json(\"tuits006.txt\", lines = True) # @M_OlgaSCordero\n",
    "tuits_7 = pd.read_json(\"tuits007.txt\", lines = True) # @m_ebrard\n",
    "tuits_8 = pd.read_json(\"tuits008.txt\", lines = True) # @galaniz\n",
    "tuits_9 = pd.read_json(\"tuits009.txt\", lines = True) # @elsaamabel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Añadir etiqueta de nombre de la personalidad\n",
    "tuits_1['name'] = \"María Luisa Albores\"\n",
    "tuits_2['name'] = \"Lucía Riojas\"\n",
    "tuits_3['name'] = \"Víctor Villalobos Arámbula \"\n",
    "tuits_4['name'] = \"Aníbal Ostoa Ortega \"\n",
    "tuits_5['name'] = \"María Merced González \"\n",
    "tuits_6['name'] = \"Olga Sánchez Cordero\"\n",
    "tuits_7['name'] = \"Marcelo Ebrard\"\n",
    "tuits_8['name'] = \"Guillermo Alaniz\"\n",
    "tuits_9['name'] = \"Elsa Amabel Landín\"\n",
    "\n",
    "# Añadir etiqueta del género de la personalidad\n",
    "tuits_1['gender'] = \"female\"\n",
    "tuits_2['gender'] = \"female\"\n",
    "tuits_3['gender'] = \"male\"\n",
    "tuits_4['gender'] = \"male\"\n",
    "tuits_5['gender'] = \"female\"\n",
    "tuits_6['gender'] = \"female\"\n",
    "tuits_7['gender'] = \"male\"\n",
    "tuits_8['gender'] = \"male\"\n",
    "tuits_9['gender'] = \"female\"\n",
    "\n",
    "# Añadir etiqueta de cargo \n",
    "tuits_1['office'] = \"executive\"\n",
    "tuits_2['office'] = \"congress\"\n",
    "tuits_3['office'] = \"executive\"\n",
    "tuits_4['office'] = \"senate\"\n",
    "tuits_5['office'] = \"senate\"\n",
    "tuits_6['office'] = \"executive\"\n",
    "tuits_7['office'] = \"executive\"\n",
    "tuits_8['office'] = \"local\"\n",
    "tuits_9['office'] = \"local\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Unir bases de datos\n",
    "tuits = pd.concat([tuits_1, tuits_2, tuits_3, tuits_4, tuits_5, tuits_6, tuits_7, tuits_8, tuits_9])\n",
    "\n",
    "tuits['name'] = tuits['name'].astype('category') # Categorizar personas\n",
    "tuits['gender'] = tuits['gender'].astype('category') # Categorizar género\n",
    "tuits['office'] = tuits['office'].astype('category') # Categorizar nivel de gobierno"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "(27588, 40)\n",
      "Index(['created_at', 'id', 'id_str', 'text', 'source', 'truncated',\n",
      "       'in_reply_to_status_id', 'in_reply_to_status_id_str',\n",
      "       'in_reply_to_user_id', 'in_reply_to_user_id_str',\n",
      "       'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place',\n",
      "       'contributors', 'is_quote_status', 'extended_tweet', 'quote_count',\n",
      "       'reply_count', 'retweet_count', 'favorite_count', 'entities',\n",
      "       'favorited', 'retweeted', 'filter_level', 'lang', 'timestamp_ms',\n",
      "       'name', 'gender', 'office', 'display_text_range', 'possibly_sensitive',\n",
      "       'retweeted_status', 'quoted_status_id', 'quoted_status_id_str',\n",
      "       'quoted_status', 'quoted_status_permalink', 'extended_entities',\n",
      "       'withheld_in_countries'],\n",
      "      dtype='object')\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 27588 entries, 0 to 13\n",
      "Data columns (total 40 columns):\n",
      " #   Column                     Non-Null Count  Dtype              \n",
      "---  ------                     --------------  -----              \n",
      " 0   created_at                 27588 non-null  datetime64[ns, UTC]\n",
      " 1   id                         27588 non-null  int64              \n",
      " 2   id_str                     27588 non-null  int64              \n",
      " 3   text                       27588 non-null  object             \n",
      " 4   source                     27588 non-null  object             \n",
      " 5   truncated                  27588 non-null  bool               \n",
      " 6   in_reply_to_status_id      7586 non-null   float64            \n",
      " 7   in_reply_to_status_id_str  7586 non-null   float64            \n",
      " 8   in_reply_to_user_id        8247 non-null   float64            \n",
      " 9   in_reply_to_user_id_str    8247 non-null   float64            \n",
      " 10  in_reply_to_screen_name    8247 non-null   object             \n",
      " 11  user                       27588 non-null  object             \n",
      " 12  geo                        6 non-null      object             \n",
      " 13  coordinates                6 non-null      object             \n",
      " 14  place                      4209 non-null   object             \n",
      " 15  contributors               0 non-null      float64            \n",
      " 16  is_quote_status            27588 non-null  bool               \n",
      " 17  extended_tweet             6131 non-null   object             \n",
      " 18  quote_count                27588 non-null  int64              \n",
      " 19  reply_count                27588 non-null  int64              \n",
      " 20  retweet_count              27588 non-null  int64              \n",
      " 21  favorite_count             27588 non-null  int64              \n",
      " 22  entities                   27588 non-null  object             \n",
      " 23  favorited                  27588 non-null  bool               \n",
      " 24  retweeted                  27588 non-null  bool               \n",
      " 25  filter_level               27588 non-null  object             \n",
      " 26  lang                       27588 non-null  object             \n",
      " 27  timestamp_ms               27588 non-null  datetime64[ns]     \n",
      " 28  name                       27588 non-null  category           \n",
      " 29  gender                     27588 non-null  category           \n",
      " 30  office                     27588 non-null  category           \n",
      " 31  display_text_range         8129 non-null   object             \n",
      " 32  possibly_sensitive         2717 non-null   object             \n",
      " 33  retweeted_status           16109 non-null  object             \n",
      " 34  quoted_status_id           6702 non-null   float64            \n",
      " 35  quoted_status_id_str       6702 non-null   float64            \n",
      " 36  quoted_status              6698 non-null   object             \n",
      " 37  quoted_status_permalink    6698 non-null   object             \n",
      " 38  extended_entities          398 non-null    object             \n",
      " 39  withheld_in_countries      3 non-null      object             \n",
      "dtypes: bool(4), category(3), datetime64[ns, UTC](1), datetime64[ns](1), float64(7), int64(6), object(18)\n",
      "memory usage: 7.3+ MB\n"
     ]
    }
   ],
   "source": [
    "# Propiedades de la base de datos\n",
    "    #print(type(tuits))\n",
    "    #print(tuits.shape)\n",
    "    #print(tuits.columns)\n",
    "    #tuits.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    @Mary_Luisa_AG buenas noches ING MARIA LUISA ALBORES \\nMIRE HACE TIEMPO QUIERO HACER CONTACTO CON NUESTRO SEÑOR PRES… https://t.co/4qUx7CGxM8\n",
       "0     @politicomx @LuRiojas @MarthaTagle @YuririaSierra @jenarovillamil @CitlaHM @warkentin Seguramente { por como se man… https://t.co/5gpIaprI5i\n",
       "0     RT @vmva1950: #ConferenciaPresidente anunciamos el arranque de #PreciosDeGarantía para pequeños productores de maíz y frijol, cosecha PV 20…\n",
       "0                                                Séptima Reunión Ordinaria de la Comisión de Energía en el @senadomexicano https://t.co/Dc0drYiUOT\n",
       "0     RT @senadomexicano: ▶ #HoyEnElSenado se realiza sesión solemne para reconocer el desempeño de atletas mexicanos que participaron en los Jue…\n",
       "0     RT @Moreno14May: @A_Encinas_R @lopezobrador_    #SinLasFamiliasNo o como era? es súper lamentable lo qué pasó con la falta de transparencia…\n",
       "0                        @m_zamarripa @m_ebrard @ONU_es Vacio el auditorio ... creo que hubo 2 o 3 menos que con la retrasada mental de Venezuela.\n",
       "0     #Entérate Propone diputado @galaniz68 polémica iniciativa para la @uaa_mx, la casa de estudios responde que no está… https://t.co/KdNbD0pgco\n",
       "0     RT @elsaamabel: A través del Centro de Justicia para las Mujeres te invitamos a la primer Jornada de empleo para Mujeres, el Lunes 7 de Oct…\n",
       "Name: text, dtype: object"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Examinar naturaleza de los tuits\n",
    "pd.set_option('display.max_colwidth', None) # Ver todo el texto\n",
    "tuits.text[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>gender</th>\n",
       "      <th>percentage</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>gender</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>female</th>\n",
       "      <td>14854</td>\n",
       "      <td>female</td>\n",
       "      <td>53.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>male</th>\n",
       "      <td>12734</td>\n",
       "      <td>male</td>\n",
       "      <td>46.2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         text  gender  percentage\n",
       "gender                           \n",
       "female  14854  female        53.8\n",
       "male    12734    male        46.2"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Gráficas descriptivas: género \n",
    "\n",
    "#Transformación de los datos\n",
    "gender_tuits = pd.DataFrame({'text' : tuits.groupby('gender')['text'].count()})\n",
    "gender_tuits['gender'] = gender_tuits.index\n",
    "gender_tuits['percentage'] = round((gender_tuits['text']/sum(gender_tuits['text']))*100, 1)\n",
    "\n",
    "\n",
    "# Construcción de la gráfica\n",
    "sequential_colors = sns.color_palette(\"RdPu\", 2)\n",
    "sns.set_palette(sequential_colors)\n",
    "#sns.set_palette('Paired')\n",
    "gender_plot = sns.barplot(x = 'gender', y ='text', data = gender_tuits)\n",
    "gender_plot.set(xlabel='Género', ylabel='Número de tuits recopilados', title = \"Distribución de tuits por género\")\n",
    "gender_plot.text(x = -0.1, y = 15000, s = \"53.8%\")\n",
    "gender_plot.text(x = 0.9, y = 13000, s = \"46.2%\")\n",
    "\n",
    "gender_tuits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>office</th>\n",
       "      <th>percentage</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>office</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>congress</th>\n",
       "      <td>908</td>\n",
       "      <td>congress</td>\n",
       "      <td>3.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>executive</th>\n",
       "      <td>26560</td>\n",
       "      <td>executive</td>\n",
       "      <td>96.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>local</th>\n",
       "      <td>18</td>\n",
       "      <td>local</td>\n",
       "      <td>0.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>senate</th>\n",
       "      <td>102</td>\n",
       "      <td>senate</td>\n",
       "      <td>0.4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            text     office  percentage\n",
       "office                                 \n",
       "congress     908   congress         3.3\n",
       "executive  26560  executive        96.3\n",
       "local         18      local         0.1\n",
       "senate       102     senate         0.4"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZIAAAEWCAYAAABMoxE0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nO3de7yVY/7/8de7ExKK4oskRuNQqiHFDKM5OJsYTBiHxmGa8cuczIzjEGKG78iQ02CGYubLNE7FNMghjRmpUApDSZRCUUTo9Pn9cV8rq93ae69ae7Vbe7+fj8d67Pu+7tN132vt9VnX4b5uRQRmZmZrq0l9Z8DMzCqbA4mZmZXEgcTMzEriQGJmZiVxIDEzs5I4kJiZWUkcSBoRSX+UdGEd7auDpI8lNU3zYySdXhf7rnKcjyXtWCWtiaQRkk6tw+MMlXRZXe2vlmPV2fuwrkn6p6R+6+A4P5D0dLmPY3WjWX1nwOqGpJnAVsAyYDnwMnAHcEtErACIiB+vwb5Oj4jHqlsnIt4CWpWW69pFRKFjXA48HhG3lfv4xZAUQKeImF7M+vnvg6TewF8ion2ZsrfWJF0M7BQRJ+bSIuKQ+suRra8cSBqW70TEY5I2A/YHrgV6AafU5UEkNYuIZXW5zzUREefV17Ebqvp+T9c3kgQo9yPMauaqrQYoIj6MiJHAsUA/SV1g1eobSW0lPSRpoaQPJP0rVRndCXQAHkzVSmdL6igpJJ0m6S3giby0/B8jX5I0XtKHqepp83Ss3pJm5+dR0kxJ307TTSWdL+l1SYskPSdpu7QsJO2UpjeTdIekeZLelPQbSU3Ssh9IelrSVZIWSHpDUrW/niV9RdLz6Xh/AzassvxwSZPS9fmPpK7V7GdsmpycrtexhaplqpzHUEmXSdoY+CewTdr2Y0nbSOopaaKkjyS9K+nqao7dW9LsdO3mp2t6Qt7y2q7XvyX9QdIHwMVV9n0wcD5wbMrX5JS+sgozbx/Xpff8v5K+lbePbSSNTJ+v6ZJ+WMP7sUVa9yNJ44EvVVm+i6TRaV+vSupbw742l3S7pDnps/BASm+TPvPzUvpDktrnbTdG0uWS/g0sBnaUtIOkselz8pikGyT9JW+bPpJeSp+TMZJ2rS5fDVpE+NUAXsBM4NsF0t8CzkjTQ4HL0vTvgD8CzdNrP7JfYKvtC+gIBFlV2cbARnlpzdI6Y4C3gS5pnXvJqmwAegOzq8sv8GtgCrAzIKAbsEVaFmTVK6TjjwA2Scd/DTgtLfsBsBT4IdAUOAOYkzunKsduAbwJ/CKd+zFp29y12QN4j6w01xTol/K7QTXXfmUe8/LydHXrVHkfCl2bZ4CT0nQrYO9qjtubrCrzamADslLoJ8DORV6vZcBPyGomNiqw/4tz72Fe2hiyas/8feSu47HAh8DmaflTwI1kQbo7MA/4VjXncjcwnOyz04Xss/R0WrYxMIusZN0svT/zgc7V7OsfwN+ANilf+6f0LYCjgZbpmvwdeKDKub0FdE7HaZ7ei6vSZ2Zf4CO++Fx/OV3vA9K6ZwPTgRb1/X2wrl8ukTR8c4DNC6QvBbYGto+IpRHxr0j/HTW4OCI+iYhPq1l+Z0RMjYhPgAuBvkqN8bU4HfhNRLwamckR8X7+Cmk/xwLnRcSiiJgJDAZOylvtzYi4NSKWA8PS+W1V4Hh7k/3jX5PO/R5gQt7yHwI3R8SzEbE8IoYBn6ft1oWlwE6S2kbExxExrpb1L4yIzyPiKbIv0b5FXq85EXFdRCyr4T2tzXt8cR3/BrwKHJZKlPsC50TEZxExCfhTleMDK9/bo4GL0udrKtn7l3M4MDMibk95fZ7sh8oxBfa1NXAI8OOIWJDy9RRARLwfEfdGxOKIWETW1rZ/lV0MjYiXIqvm2xrYK+VrSUQ8DYzMW/dY4B8RMToilpIFnI2Ar67JBWwIHEgavm2BDwqk/57s19OjkmZIOreIfc1ag+Vvkn1Zty1iv9sBr9eyTlu+KEnkH2PbvPl3chMRsThNFmqs3wZ4u0rgzN/v9sAvU3XFQkkLUx63qSWPdeU0sl+7/5U0QdLhNay7IAXunDfJ8lnM9art/SxGoeu4TXp9kL6wqzt+TjuyEkDVz0/O9kCvKu/HCcD/FNjXdum4C6oukNRS0s2pmu8jYCzQusqPnfw85M5hcQ3LV+YzsvaUWdWcY4PmQNKASdqL7EO9WjfK9Cv1lxGxI/Ad4Ky8+u3qSia1lVi2y5vuQPbLej5Z8b9lXr6akn155MyiSp14AfPT/ravcoy3a9mukLnAtpJUZV/5+bk8IlrnvVpGxF1F7r/q+Rb6wstZ7ZpGxLSIOB7YErgSuCe1pxTSpsqyDmSl0GKuV23vZzFDgxe6jnPSa3NJm9Rw/Jx5ZFVkVT8/ObOAp6q8H60i4owC+5qVjtu6wLJfklWf9oqITYGvp/T8/Oef89y0r5Z5afl5nEPe9U3XYbtqzrFBcyBpgCRtmn7F3k1WnzulwDqHS9opffg/IusyvDwtfhfYseo2RThR0m7pH+9S4J5UzfQasKGkwyQ1B35DVqef8ydgkKROynSVtEX+jtN+hgOXS9pE0vbAWcBfWHPPkH1x/VRSM0lHAT3zlt8K/FhSr5SfjVPeNym4t9Wv12Sgs6TukjakSkN2gW23UNbTDgBJJ0pql37hLkzJywtunblEUgtJ+5FVA/29jq7Xu0BHpQb6amxJdh2bS/oesCswKiJmAf8BfidpQ2WdFU4D/lp1Bymv9wEXp1LDbmTtUjkPAV+WdFI6TnNJexVq2I6IuWQdGG5MjevNJeUCxibAp8BCZR1BBtZ08hHxJjAx5auFpH3IfnTlDCerxvtW+lz/kqwK9D817bchciBpWB6UtIjsV9kFZI2w1XX97QQ8BnxM9sV6Y0SMSct+B/wmVSP8ag2OfydZQ/I7ZA2sP4WsFxnw/8gCxttkv9jze3FdTfZP+ShZUPszWV1zVT9J284gK2X9H7DG95JExBLgKLLG4gVkdd335S2fSNZOcn1aPj2tW52LgWHpevWNiNfIAuljwDQKlAjzjvVf4C5gRtp+G+Bg4CVJH5N14T4uIj6rZhfvpDzOIfuS/nHaJ5R+vf6e/r4v6flq1nmW7LM0n6zN4Zi89q3jyRr55wD3AwMjYnQ1+zmTrBryHbLP0O25Bal67EDguLSvd8hKahustpfMSWSlsf+SteH8PKVfQ/a5mg+MAx6uZvt8JwD7AO8Dl5E14n+e8vUqcCJwXdrnd8i64C8pYr8NSq6XjplVGNXzzYySfkDWg2vf+jh+fVDWVfy/EVFjaaaxcYnEzKwaqQrtS8rusToYOAJ4oL7ztb7xne1mZtX7H7Jqzy3IqmPPiIgX6jdL6x9XbZmZWUlctWVmZiVpdFVbbdu2jY4dO9Z3NszMKspzzz03PyLaFVrW6AJJx44dmThxYn1nw8ysokh6s7plrtqyinHttdfSpUsXOnfuzDXXXLMy/brrrmPnnXemc+fOnH322att99lnn9GzZ0+6detG586dGTjwi56bJ5xwAl27duX8889fmTZo0CBGjBhR3pMxa0AaXYnEKtPUqVO59dZbGT9+PC1atODggw/msMMOY/bs2YwYMYIXX3yRDTbYgPfee2+1bTfYYAOeeOIJWrVqxdKlS9l333055JBDaNkyG/nixRdfZL/99uPDDz9k8eLFjB8/ngsvrMgHGJrVCwcSqwivvPIKe++998ov//3335/777+fiRMncu6557LBBtlNzltuueVq20qiVats7MalS5eydOlSJNG8eXM+/fRTVqxYwZIlS2jatCkXXXQRl1566bo7MbMGwFVbVhG6dOnC2LFjef/991m8eDGjRo1i1qxZvPbaa/zrX/+iV69e7L///kyYMKHg9suXL6d79+5sueWWHHDAAfTq1Ytdd92VDh06sMcee9C3b1+mT59ORPCVr3xlHZ+dWWVzicQqwq677so555zDAQccQKtWrejWrRvNmjVj2bJlLFiwgHHjxjFhwgT69u3LjBkzWHVAWmjatCmTJk1i4cKFfPe732Xq1Kl06dJllbaW73znO9x8881cfvnlTJ48mQMOOIAf/rDah/qZWeISiVWM0047jeeff56xY8ey+eab06lTJ9q3b89RRx2FJHr27EmTJk2YP39+tfto3bo1vXv35uGHVx2vb8SIEfTo0YNPPvmEqVOnMnz4cO68804WL15czZ7MLMeBxCpGriH9rbfe4r777uP444/nyCOP5IknngDgtddeY8mSJbRtu+qztObNm8fChdlo7J9++imPPfYYu+yyy8rlS5cu5dprr+XXv/41ixcvXlmaybWdmFnNXLVlFePoo4/m/fffp3nz5txwww20adOGU089lVNPPZUuXbrQokULhg0bhiTmzJnD6aefzqhRo5g7dy79+vVj+fLlrFixgr59+3L44V88dPCGG26gX79+tGzZkq5duxIR7L777hx66KG0bl3o+Uhmlq/RjbXVo0eP8A2JZmZrRtJzEdGj0DKXSKysbr7zoPrOwnrjRyc9Ut9ZMCsLt5GYmVlJHEjMzKwkDiRmZlYSBxIzMyuJA4mZmZXEgcTMzEriQGJmZiVxIDEzs5KULZBI2k7Sk5JekfSSpJ+l9IslvS1pUnodmrfNeZKmS3pV0kF56QentOmSzs1L30HSs5KmSfqbpBblOh8zMyusnCWSZcAvI2JXYG9ggKTd0rI/RET39BoFkJYdB3QGDgZulNRUUlPgBuAQYDfg+Lz9XJn21QlYAJxWxvMxM7MCyhZIImJuRDyfphcBrwDb1rDJEcDdEfF5RLwBTAd6ptf0iJgREUuAu4EjlA3R+k3gnrT9MODI8pyNmZlVZ520kUjqCHwFeDYlnSnpRUm3SWqT0rYFZuVtNjulVZe+BbAwIpZVSS90/P6SJkqaOG/evDo4IzMzyyl7IJHUCrgX+HlEfATcBHwJ6A7MBQbnVi2weaxF+uqJEbdERI+I6NGuXbs1PAMzM6tJWUf/ldScLIj8NSLuA4iId/OW3wo8lGZnA9vlbd4emJOmC6XPB1pLapZKJfnrm5nZOlLOXlsC/gy8EhFX56Vvnbfad4GpaXokcJykDSTtAHQCxgMTgE6ph1YLsgb5kZE9SOVJ4Ji0fT9gRLnOx8zMCitnieRrwEnAFEmTUtr5ZL2uupNVQ80EfgQQES9JGg68TNbja0BELAeQdCbwCNAUuC0iXkr7Owe4W9JlwAtkgcvMzNahsgWSiHiawu0Yo2rY5nLg8gLpowptFxEzyHp1mZlZPfGd7WZmVhIHEjMzK4kDiZmZlcSBxMzMSuJAYmZmJXEgMTOzkjiQmJlZSRxIzMysJA4kZmZWEgcSMzMriQOJmZmVxIHEzMxKUlQgkfQzSZsq82dJz0s6sNyZMzOz9V+xJZJT09MNDwTaAacAV5QtV2ZmVjGKDSS54eAPBW6PiMkUHiLezMwamWIDyXOSHiULJI9I2gRYUb5smZlZpSj2wVanAd2BGRGxWNIWZNVbZmbWyBUVSCJihaT2wPezR7HzVEQ8WNacmZlZRSi219YVwM/Inqf+MvBTSb8rZ8bMzKwyFFu1dSjQPSJWAEgaBrwAnFeujJmZWWVYkxsSW+dNb1bXGTEzs8pUbInkd8ALkp4k6/b7dVwaMTMzim9sv0vSGGAvskByTkS8U86MmZlZZagxkEjao0rS7PR3G0nbRMTz5cmWmZlVitpKJIPT3w2BHkDujvauwLPAvuXLmpmZVYIaG9sj4hsR8Q3gTWCPiOgREXsCXwGmr4sMmpnZ+q3YXlu7RMSU3ExETCW7093MzBq5YnttvSLpT8BfgABOBF4pW67MzKxiFBtITgHOILu7HWAscFNZcmRmZhWl2O6/nwF/SC8zM7OVih1rq5OkeyS9LGlG7lXEdttJelLSK5JekvSzlL65pNGSpqW/bVK6JA2RNF3Si/ndjyX1S+tPk9QvL31PSVPSNkOURpU0M7N1o9jG9tvJqrKWAd8A7gDuLGK7ZcAvI2JXYG9ggKTdgHOBxyOiE/B4mgc4BOiUXv3TMZG0OTAQ6AX0BAbmgk9ap3/edgcXeU5mZlYHig0kG0XE44Ai4s2IuBj4Zm0bRcTc3E2LEbGIrIF+W+AIYFhabRhwZJo+ArgjMuOA1pK2Bg4CRkfEBxGxABgNHJyWbRoRz0REkAW43L7MzGwdKLax/TNJTYBpks4E3ga2XJMDSepIdv/Js8BWETEXsmAjKbevbYFZeZvNTmk1pc8ukF712P3JSi106NBhTbJtZma1KLZE8nOgJfBTYE/gJKBfjVvkkdQKuBf4eUR8VNOqBdJiLdJXTYi4Jd1M2aNdu3bFZNnMzIpUbK+tCWnyY9bwEbuSmpMFkb9GxH0p+V1JW6fSyNbAeyl9NrBd3ubtgTkpvXeV9DEpvX2B9c3MbB2pbdDGBynwCz8nIvrUsr2APwOvRMTVeYtGkpVorkh/R+SlnynpbrKG9Q9TsHkE+G1eA/uBwHkR8YGkRZL2JqsyOxm4rqY8mZlZ3aqtRHJVifv/Glk12BRJk1La+WQBZLik04C3gO+lZaPInsY4HVhMKv2kgDEIyJWMLo2ID9L0GcBQYCPgn+llZmbrSI2BJCKeKmXnEfE0hdsxAL5VYP0ABlSzr9uA2wqkTwS6lJBNMzMrQW1VW8Mjoq+kKaxaxSWy7/2uZc2dmZmt92qr2sqNrXV4uTNiZmaVqbbnkeTu9XgT+BzoRvZQq89TmpmZNXLFjrV1OjAeOAo4Bhgn6dRyZszMzCpDsXe2/xr4SkS8DyBpC+A/FGj8NjOzxqXYO9tnA4vy5hex6pAlZmbWSBVbInkbeFbSCLLeW0cA4yWdBVDlZkMzM2tEig0kr6dXTu5O9E3qNjtmZlZpih1r6xIASZtks/FxWXNlZmYVo9heW10kvQBMBV6S9JykzuXNmpmZVYJiG9tvAc6KiO0jYnvgl8Ct5cuWmZlVimIDycYR8WRuJiLGABuXJUdmZlZRim1snyHpQr54TvuJwBvlyZKZmVWSYkskpwLtgPvSqy1r+IArMzNrmIrttbWA7DG7ZmZmqyi219ZoSa3z5tukpxaamVkjV2zVVtuIWJibSSWULcuTJTMzqyTFBpIVkjrkZiRtTw3Pcjczs8aj2F5bFwBPS8o9evfrQP/yZMnMzCpJsY3tD0vaA9ib7DG7v4iI+WXNmZmZVYRiG9sFHAzsEREPAi0l9SxrzszMrCIU20ZyI7APcHyaXwTcUJYcmZlZRSm2jaRXROyRBm4kIhZIalHGfJmZWYUotkSyVFJTUk8tSe2AFWXLlZmZVYxiA8kQ4H5gS0mXA08Dvy1brszMrGIU22vrr5KeA75F1mvryIh4paw5MzOzilBrIJHUBHgxIroA/y1/lszMrJLUWrUVESuAyfl3tpuZmeUU22tra7JH7I4HPsklRkSfsuTKzMwqRrGB5JKy5sLMzCpWUb22IuKpQq/atpN0m6T3JE3NS7tY0tuSJqXXoXnLzpM0XdKrkg7KSz84pU2XdG5e+g6SnpU0TdLffG+Lmdm6V2z337U1lGxolar+EBHd02sUgKTdgOOAzmmbGyU1Tfev3AAcAuwGHJ/WBbgy7asTsAA4raxnY2ZmqylrIImIscAHRa5+BHB3RHweEW8A04Ge6TU9ImZExBLgbuCINP7XN4F70vbDgCPr9ATMzKxWaxxI0tMRu5Z43DMlvZiqvtqktG2BWXnrzE5p1aVvASyMiGVV0gvlub+kiZImzps3r8Ssm5lZvmJH/x0jaVNJmwOTgdslXb2Wx7wJ+BLQHZgLDM4dpsC6sRbpqydG3BIRPSKiR7t27dY8x2ZmVq1iSySbRcRHwFHA7RGxJ/DttTlgRLwbEcvT/Sm3klVdQVai2C5v1fbAnBrS5wOtJTWrkm5mZutQsYGkmaStgb7AQ6UcMO0n57tArkfXSOA4SRtI2gHoBIwHJgCdUg+tFmQN8iMjIoAngWPS9v2AEaXkzczM1tya3EfyCPB0REyQtCMwrbaNJN0F9AbaSpoNDAR6S+pOVg01E/gRQES8JGk48DKwDBgQEcvTfs5Mx28K3BYRL6VDnAPcLeky4AXgz0Wej5mZ1ZFiA8nciFjZwB4RM4ppI4mI4wskV/tlHxGXA5cXSB8FjCqQPoMvqsbMzKweFFu1dV2RaWZm1sjUWCKRtA/wVaCdpLPyFm1KVs1kZmaNXG1VWy2AVmm9TfLSP+KLRm4zM2vEagwkaTytpyQNjYg311GezMysgtRWtXVNRPwcuF7Sajf7eRh5MzOrrWrrzvT3qnJnxMzMKlNtVVvPpb+1DhlvZmaNU1H3kUh6gwLjWEXEjnWeIzMzqyjF3pDYI296Q+B7wOZ1nx0zM6s0xT4h8f2819sRcQ3Zs0DMzKyRK7Zqa4+82SZkJZRNqlndzMwakWKrtgbnTS8D3iAbCdjMzBq5YgPJaWmAxJXSUO9mZtbIFTto4z1FppmZWSNT253tuwCdgc0kHZW3aFOy3ltmZtbI1Va1tTNwONAa+E5e+iLgh+XKlJmZVY7a7mwfAYyQtE9EPLOO8mRmZhWk2PtIHETMzKygYhvbzczMCnIgMTOzkhQVSCRtJenPkv6Z5neTdFp5s2ZmZpWg2BLJUOARYJs0/xrw83JkyMzMKkuxgaRtRAwHVgBExDJgedlyZWZmFaPYQPKJpC1IzySRtDfwYdlyZWZmFaPYsbbOAkYCX5L0b6AdcEzZcmVmZhWjqEASEc9L2p/sTncBr0bE0rLmzMzMKkJtY20dVc2iL0siIu4rQ57MzKyC1FYiyY2vtSXwVeCJNP8NYAzgQGJm1sjVNtbWKQCSHgJ2i4i5aX5r4IbyZ8/MzNZ3xfba6pgLIsm7wJfLkB8zM6swxQaSMZIekfQDSf2AfwBP1raRpNskvSdpal7a5pJGS5qW/rZJ6ZI0RNJ0SS/mPydeUr+0/rR0/Fz6npKmpG2GSFLRZ25mZnWi2NF/zwT+CHQDugO3RMRPith0KHBwlbRzgccjohPweJoHOATolF79gZsgCzzAQKAX0BMYmAs+aZ3+edtVPZaZmZVZsfeREBH3A/evyc4jYqykjlWSjwB6p+lhZI3256T0OyIigHGSWqe2mN7A6Ij4AEDSaOBgSWOATXND3Eu6AzgS+Oea5NHMzEpTH6P/bpVrb0l/t0zp2wKz8tabndJqSp9dIH01kvpLmihp4rx58+rkJMzMLLM+DSNfqH0j1iJ99cSIWyKiR0T0aNeuXQlZNDOzqooOJJJaSOqSXs1LOOa7qcoq1434vZQ+G9gub732wJxa0tsXSDczs3Wo2OeR9Aamkd07ciPwmqSvr+UxRwK5nlf9gBF56Sen3lt7Ax+mqq9HgAMltUmN7AcCj6RliyTtnXprnZy3LzMzW0eKbWwfDBwYEa8CSPoycBewZ00bSbqLrLG8raTZZL2vrgCGpwdjvQV8L60+CjgUmA4sBk4BiIgPJA0CJqT1Ls01vANnkPUM24iskd0N7WZm61ixgaR5LogARMRrxVRvRcTx1Sz6VoF1AxhQzX5uA24rkD4R6FJbPszMrHyKDSQTJf0ZuDPNnwA8V54smZlZJSk2kJxBVlr4KVlvqbFkbSVmZtbI1RpIJDUF/hwRJwJXlz9LZmZWSWrttRURy4F2klqsg/yYmVmFKbZqaybwb0kjgU9yiRHhEoqZWSNXbCCZk15NgE3Klx0zM6s0xT6z/RIASRtHxCe1rW9mZo1HsXe27yPpZeCVNN9NknttmZlZ0WNtXQMcBLwPEBGTgbUdIsXMzBqQogdtjIhZVZKW13FezMysAhXb2D5L0leBSN2Af0qq5jIzs8at2BLJj8nubM89TKo71YyLZWZmjUuxvbbmk42vZWZmtoqiAomkHYCfAB3zt4mIPuXJlpmZVYpi20geAP4MPAisKF92zMys0hQbSD6LiCFlzYmZmVWkYgPJtZIGAo8Cn+cSI+L5suTKzMwqRrGBZHfgJOCbfFG1FWnezMwasWIDyXeBHSNiSTkzY2ZmlafY+0gmA63LmREzM6tMxZZItgL+K2kCq7aRuPuvmVkjV2wgGVjWXJiZWcUq9s72p8qdETMzq0zVBhJJLSNicZpeRNZLC6AF0Bz4JCI2LX8WzcxsfVZTieQHktpExOURscrjdSUdCfQsb9bMzKwSVNtrKyJuBN6UdHKBZQ/ge0jMzIxa2kgi4i8Ako7KS24C9OCLqi4zM2vEiu219Z286WXATOCIOs+NmZlVnGJ7bZ1S7oyYmVllqjGQSLqohsUREYPqOD9mZlZhahsi5ZMCL4DTgHNKObCkmZKmSJokaWJK21zSaEnT0t82KV2ShkiaLulFSXvk7adfWn+apH6l5MnMzNZcbY3tg3PTkjYBfgacAtwNDK5uuzXwjfQY35xzgccj4gpJ56b5c4BDgE7p1Qu4CeglaXOyu+5zjf/PSRoZEQvqIG9mZlaEWgdtTKWEy4AXyQLPHhFxTkS8V4b8HAEMS9PDgCPz0u+IzDigtaStgYOA0RHxQQoeo4GDy5AvMzOrRo2BRNLvgQnAImD3iLi4Dn/tB/CopOck9U9pW0XEXID0d8uUvi0wK2/b2SmtuvSq59Ff0kRJE+fNm1dH2TczM6i919YvyUb7/Q1wgaRcusga20sZIuVrETFH0pbAaEn/rWFdFUiLGtJXTYi4BbgFoEePHr7/xcysDtXWRlLs80rWWETMSX/fk3Q/2ZAr70raOiLmpqqrXPXZbGC7vM3bA3NSeu8q6WPKlWczM1td2QJFTSRtnBrvkbQxcCAwFRgJ5Hpe9QNGpOmRwMmp99bewIep6usR4EBJbVIPrwNTmpmZrSPF3tle17YC7k9VZc2A/4uIh9ODs4ZLOg14C/heWn8UcCgwHVhM1nOMiPhA0iCydhyASyPig3V3GmZmVi+BJCJmAN0KpL8PfKtAegADqtnXbcBtdZ1HMzMrTr1UbZmZWcPhQGJmZiVxIDEzs5I4kJiZWUkcSMzMrCQOJGZmVhIHEjMzK4kDiZmZlcSBxMzMSuJAYmZmJXEgMTOzkjiQmJlZSRxIzMysJA4kZmZWEgcSMzMriQOJmZmVxIHEzNEZYaEAAA/ISURBVMxK4kBSTz777DN69uxJt27d6Ny5MwMHDlxtnT/+8Y/svvvudO/enX333ZeXX34ZgH//+9907dqVvfbai+nTpwOwcOFCDjroILKHSZqZrTsOJPVkgw024IknnmDy5MlMmjSJhx9+mHHjxq2yzve//32mTJnCpEmTOPvssznrrLMAGDx4MPfeey+//e1vuemmmwAYNGgQ559/PpLW+bmYWePmQFJPJNGqVSsAli5dytKlS1cLAptuuunK6U8++WTl8ubNm/Ppp5+yePFimjdvzuuvv87bb7/N/vvvv+5OwMwsaVbfGWjMli9fzp577sn06dMZMGAAvXr1Wm2dG264gauvvpolS5bwxBNPAHDeeefRv39/NtpoI+68805+9atfMWjQoHWdfTMzwCWSetW0aVMmTZrE7NmzGT9+PFOnTl1tnQEDBvD6669z5ZVXctlllwHQvXt3xo0bx5NPPsmMGTPYZpttiAiOPfZYTjzxRN599911fSpm1og5kKwHWrduTe/evXn44YerXee4447jgQceWCUtIrjsssu48MILueSSS7jkkks48cQTGTJkSLmzbGa2kgNJPZk3bx4LFy4E4NNPP+Wxxx5jl112WWWdadOmrZz+xz/+QadOnVZZPmzYMA477DDatGnD4sWLadKkCU2aNGHx4sXlPwFr8B5++GF23nlndtppJ6644orVlo8dO5Y99tiDZs2acc8996xMf/XVV9lzzz3p1q0bzzzzDADLli3j29/+dqP9bNZ2LXPuueceJDFx4kSgcnpouo2knsydO5d+/fqxfPlyVqxYQd++fTn88MO56KKL6NGjB3369OH666/nscceo3nz5rRp04Zhw4at3H7x4sUM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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Gráficas descriptivas: office \n",
    "#Transformación de los datos\n",
    "office_tuits = pd.DataFrame({'text' : tuits.groupby('office')['text'].count()})\n",
    "office_tuits['office'] = office_tuits.index\n",
    "office_tuits['percentage'] = round((office_tuits['text']/sum(office_tuits['text']))*100, 1)\n",
    "\n",
    "\n",
    "# Contrucción de la gráfica\n",
    "sequential_colors = sns.color_palette(\"RdPu\", 2)\n",
    "sns.set_palette(sequential_colors)\n",
    "office_plot = sns.barplot(x = 'office', y ='text', data = office_tuits)\n",
    "office_plot.set(xlabel='Nivel de gobierno', ylabel='Número de tuits recopilados', title = \"Distribución de tuits por tipo de cargo\")\n",
    "\n",
    "# Etiquetas de porcentajes \n",
    "office_plot.text(x = -0.1, y = 1100, s = \"3.3%\")\n",
    "office_plot.text(x = 0.8, y = 26700, s = \"96.3%\")\n",
    "office_plot.text(x = 1.9, y = 300, s = \"0.1%\")\n",
    "office_plot.text(x = 2.9, y = 300, s = \"0.4%\")\n",
    "\n",
    "office_tuits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>name</th>\n",
       "      <th>percentage</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>name</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Aníbal Ostoa Ortega</th>\n",
       "      <td>92</td>\n",
       "      <td>Aníbal Ostoa Ortega</td>\n",
       "      <td>0.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Elsa Amabel Landín</th>\n",
       "      <td>14</td>\n",
       "      <td>Elsa Amabel Landín</td>\n",
       "      <td>0.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Guillermo Alaniz</th>\n",
       "      <td>4</td>\n",
       "      <td>Guillermo Alaniz</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Lucía Riojas</th>\n",
       "      <td>908</td>\n",
       "      <td>Lucía Riojas</td>\n",
       "      <td>3.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Marcelo Ebrard</th>\n",
       "      <td>12174</td>\n",
       "      <td>Marcelo Ebrard</td>\n",
       "      <td>44.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>María Luisa Albores</th>\n",
       "      <td>1</td>\n",
       "      <td>María Luisa Albores</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>María Merced González</th>\n",
       "      <td>10</td>\n",
       "      <td>María Merced González</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Olga Sánchez Cordero</th>\n",
       "      <td>13921</td>\n",
       "      <td>Olga Sánchez Cordero</td>\n",
       "      <td>50.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Víctor Villalobos Arámbula</th>\n",
       "      <td>464</td>\n",
       "      <td>Víctor Villalobos Arámbula</td>\n",
       "      <td>1.7</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                              text                         name  percentage\n",
       "name                                                                       \n",
       "Aníbal Ostoa Ortega             92         Aníbal Ostoa Ortega          0.3\n",
       "Elsa Amabel Landín              14           Elsa Amabel Landín         0.1\n",
       "Guillermo Alaniz                 4             Guillermo Alaniz         0.0\n",
       "Lucía Riojas                   908                 Lucía Riojas         3.3\n",
       "Marcelo Ebrard               12174               Marcelo Ebrard        44.1\n",
       "María Luisa Albores              1          María Luisa Albores         0.0\n",
       "María Merced González           10       María Merced González          0.0\n",
       "Olga Sánchez Cordero         13921         Olga Sánchez Cordero        50.5\n",
       "Víctor Villalobos Arámbula     464  Víctor Villalobos Arámbula          1.7"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Gráficas descriptivas: personas \n",
    "\n",
    "#Transformación de los datos\n",
    "name_tuits = pd.DataFrame({'text' : tuits.groupby('name')['text'].count()})\n",
    "name_tuits['name'] = name_tuits.index\n",
    "name_tuits['percentage'] = round((name_tuits['text']/sum(name_tuits['text']))*100, 1)\n",
    "\n",
    "\n",
    "# Construcción de la gráfica\n",
    "sequential_colors = sns.color_palette(\"RdPu\", 2)\n",
    "sns.set_palette(sequential_colors)\n",
    "name_plot = sns.barplot(x = 'name', y ='text', data = name_tuits)\n",
    "name_plot.set(xlabel='Persona', ylabel='Número de tuits recopilados', title = \"Distribución de tuits por persona\")\n",
    "\n",
    "name_tuits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5520, 40)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Seleccionar muestra para etiquetar\n",
    "tuits_sample = tuits.sample(5520)\n",
    "tuits_sample.to_csv(\"tuits_train.csv\", encoding = 'utf-8')\n",
    "\n",
    "# Todos los tuits de esta selección aleatoria serán clasificados manualmente\n",
    "tuits_sample.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(5520, 6)\n",
      "text          object\n",
      "class       category\n",
      "gendered    category\n",
      "name        category\n",
      "gender      category\n",
      "office      category\n",
      "dtype: object\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>class</th>\n",
       "      <th>gendered</th>\n",
       "      <th>name</th>\n",
       "      <th>gender</th>\n",
       "      <th>office</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>text</td>\n",
       "      <td>neutral</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Olga Sánchez Cordero</td>\n",
       "      <td>female</td>\n",
       "      <td>executive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>@PlasticoTieso @LuRiojas Nos vale mucha verga hija de tu puta madre que seas lo que seas, pinche feminazi de mierda.</td>\n",
       "      <td>aggresive</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Lucía Riojas</td>\n",
       "      <td>female</td>\n",
       "      <td>congress</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>RT @veronicagmiceli: #RuthOlveraporencimadelaley\\n@alfredodelmazo \\nEn Atizapán se viola la ley \\n\\nhttps://t.co/7Uqs3QjZjG\\n@delfinagomeza  @Ir</td>\n",
       "      <td>neutral</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Olga Sánchez Cordero</td>\n",
       "      <td>female</td>\n",
       "      <td>executive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>@M_OlgaSCordero @SEGOB_mx @LuisCardenasMx @Mx_Diputados @SCJN @LuRiojas Necesitamos tener una migración regulada y",
       " https://t.co/V8h0azl4lL</td>\n",
       "      <td>neutral</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Olga Sánchez Cordero</td>\n",
       "      <td>female</td>\n",
       "      <td>executive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>RT @quebecq: @adrifadi @lopezobrador_ @HectorAstudillo C.c.p. @M_OlgaSCordero @SSPCMexico @emoctezumab @SEP_mx @tatclouthier\\n\\nSeñoras y Señ</td>\n",
       "      <td>neutral</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Olga Sánchez Cordero</td>\n",
       "      <td>female</td>\n",
       "      <td>executive</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                                                                                                                text  \\\n",
       "0                                                                                                                                               text   \n",
       "1                               @PlasticoTieso @LuRiojas Nos vale mucha verga hija de tu puta madre que seas lo que seas, pinche feminazi de mierda.   \n",
       "2  RT @veronicagmiceli: #RuthOlveraporencimadelaley\\n@alfredodelmazo \\nEn Atizapán se viola la ley \\n\\nhttps://t.co/7Uqs3QjZjG\\n@delfinagomeza  @Ir\n",
       "   \n",
       "3        @M_OlgaSCordero @SEGOB_mx @LuisCardenasMx @Mx_Diputados @SCJN @LuRiojas Necesitamos tener una migración regulada y\n",
       " https://t.co/V8h0azl4lL   \n",
       "4     RT @quebecq: @adrifadi @lopezobrador_ @HectorAstudillo C.c.p. @M_OlgaSCordero @SSPCMexico @emoctezumab @SEP_mx @tatclouthier\\n\\nSeñoras y Señ\n",
       "   \n",
       "\n",
       "       class gendered                  name  gender     office  \n",
       "0    neutral      0.0  Olga Sánchez Cordero  female  executive  \n",
       "1  aggresive      1.0          Lucía Riojas  female   congress  \n",
       "2    neutral      0.0  Olga Sánchez Cordero  female  executive  \n",
       "3    neutral      0.0  Olga Sánchez Cordero  female  executive  \n",
       "4    neutral      0.0  Olga Sánchez Cordero  female  executive  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Cargamos los datos etiquetados \n",
    "tuits_labeled = pd.read_csv(\"tuits_labeled.csv\", encoding = \"latin\")\n",
    "\n",
    "# Seleccionamos variables de interés\n",
    "tuits_labeled = tuits_labeled[['text', 'class', 'gendered', 'name', 'gender', 'office']]\n",
    "\n",
    "# Cambiamos las variables a variables categóricas\n",
    "tuits_labeled['name'] = tuits_labeled['name'].astype('category') # Categorizar personas\n",
    "tuits_labeled['gender'] = tuits_labeled['gender'].astype('category') # Categorizar género\n",
    "tuits_labeled['office'] = tuits_labeled['office'].astype('category') # Categorizar nivel de gobierno\n",
    "tuits_labeled['class'] = tuits_labeled['class'].astype('category') # Categorizar personas\n",
    "tuits_labeled['gendered'] = tuits_labeled['gendered'].astype('category') # Categorizar personas\n",
    "\n",
    "# Estudiamos las características de los datos\n",
    "print(tuits_labeled.shape)\n",
    "print(tuits_labeled.dtypes)\n",
    "tuits_labeled.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>gender</th>\n",
       "      <th>percentage</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>gender</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>female</th>\n",
       "      <td>3032</td>\n",
       "      <td>female</td>\n",
       "      <td>54.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>male</th>\n",
       "      <td>2488</td>\n",
       "      <td>male</td>\n",
       "      <td>45.1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        text  gender  percentage\n",
       "gender                          \n",
       "female  3032  female        54.9\n",
       "male    2488    male        45.1"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Gráficas descriptivas de los tuits etiquetados\n",
    "gender_tuits = pd.DataFrame({'text' : tuits_labeled.groupby('gender')['text'].count()})\n",
    "gender_tuits['gender'] = gender_tuits.index\n",
    "gender_tuits['percentage'] = round((gender_tuits['text']/sum(gender_tuits['text']))*100, 1)\n",
    "\n",
    "# Construcción de la gráfica\n",
    "sequential_colors = sns.color_palette(\"RdPu\", 2)\n",
    "sns.set_palette(sequential_colors)\n",
    "gender_plot = sns.barplot(x = 'gender', y ='text', data = gender_tuits)\n",
    "gender_plot.set(xlabel='Género', ylabel='Número de tuits recopilados', title = \"Distribución de tuits por género en muestra aleatoria\")\n",
    "\n",
    "gender_plot.text(x =-0.1, y = 3050, s = \"54.9%\")\n",
    "gender_plot.text(x = 0.9, y = 2550, s = \"45.1%\")\n",
    "\n",
    "gender_tuits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>office</th>\n",
       "      <th>percentage</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>office</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>congress</th>\n",
       "      <td>186</td>\n",
       "      <td>congress</td>\n",
       "      <td>3.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>executive</th>\n",
       "      <td>5316</td>\n",
       "      <td>executive</td>\n",
       "      <td>96.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>local</th>\n",
       "      <td>4</td>\n",
       "      <td>local</td>\n",
       "      <td>0.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>senate</th>\n",
       "      <td>14</td>\n",
       "      <td>senate</td>\n",
       "      <td>0.3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           text     office  percentage\n",
       "office                                \n",
       "congress    186   congress         3.4\n",
       "executive  5316  executive        96.3\n",
       "local         4      local         0.1\n",
       "senate       14     senate         0.3"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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p5+mR48e7LrrrkyfPh2ABQsWsN9++5H16G5mVj+cLMqsdevWPP7440yePJkXXniBBx98kHHjxq1Wb+HChVx77bX07dt3RdlVV13F3Xffza9+9StuvPFGAIYOHcr555+PVN2jyc3MysPJoswk0bZtWwCWLFnCkiVLqv2iv/DCCzn77LNZZ511VpS1bNmSzz77jEWLFtGyZUv+85//8Pbbb7PXXnvVW/xmZuBkUS+WLVtGr1696NixI/vss88qRw8Azz//PLNmzeLggw9epfy8885j8ODBXH311Zx++ulccMEFDB06tD5DNzMDnCzqRUeWAuoAAA5jSURBVPPmzXnhhReYPXs248ePZ+rUqSumLV++nDPPPJOrrrpqtfl69erFuHHjeOKJJ5gxYwadOnUiIjjmmGMYOHAg7777bn1uhpk1YU4W9ahdu3b069ePBx98cEXZwoULmTp1Kv369aNLly6MGzeOQw89dMVJboCI4LLLLuPCCy/kkksu4ZJLLmHgwIFce+21DbEZZtYEOVmU2bx581iwYAEAn332GY8++ijbb7/9iukbbrgh77//PjNnzmTmzJnstttujB49mt69e6+oM2zYMA466CDat2/PokWLaNasGc2aNWPRokX1vj1m1jS1aOgAGru5c+cyaNAgli1bxvLly+nfvz8HH3wwF110Eb179+bQQw+tdf5FixYxbNgwHn74YQDOOussjjzySFq1asWdd95ZH5tgZuYji3Lr0aMHzz//PFOmTGHq1KlcdNFFAFx66aXVJooxY8asclTRpk0bnnjiCVq2bAnAnnvuyYsvvsikSZPYdttt62cjrNF68MEH2W677dhmm2244oorVps+duxYdt55Z1q0aMHIkSNXlL/22mvssssu9OzZk2eeeQaApUuX8r3vfa9JH/GW2p8Fa+M9VU4WZk3UsmXLOO200/jnP//Jyy+/zJ133snLL7+8Sp0tt9yS22+/nR/84AerlN90001cccUVjBw5kiuvvBKAG2+8keOPP542bdrU2zZUkjz7E9bee6qcLMyaqPHjx7PNNtuw9dZb06pVK4499lhGjRq1Sp0uXbrQo0cPmjVb9aui6j1ACxYs4L777uOEE06oz02oKHn2J6y991T5nIVZE/X222+zxRZbrBjv3Lkzzz77bK55TzvtNE444QS++OILbrrpJi699FIuuOCC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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Transformación de los datos\n",
    "office_tuits = pd.DataFrame({'text' : tuits_labeled.groupby('office')['text'].count()})\n",
    "office_tuits['office'] = office_tuits.index\n",
    "office_tuits['percentage'] = round((office_tuits['text']/sum(office_tuits['text']))*100, 1)\n",
    "\n",
    "# Contrucción de la gráfica\n",
    "sequential_colors = sns.color_palette(\"RdPu\", 2)\n",
    "sns.set_palette(sequential_colors)\n",
    "office_plot = sns.barplot(x = 'office', y ='text', data = office_tuits)\n",
    "office_plot.set(xlabel='Nivel de gobierno', ylabel='Número de tuits recopilados', title = \"Distribución de tuits por tipo de cargo\")\n",
    "\n",
    "# Etiquetas de porcentajes \n",
    "office_plot.text(x = -0.1, y = 300, s = \"3.4%\")\n",
    "office_plot.text(x = 0.8, y = 5350, s = \"96.3%\")\n",
    "office_plot.text(x = 1.9, y = 100, s = \"0.1%\")\n",
    "office_plot.text(x = 2.9, y = 100, s = \"0.4%\")\n",
    "\n",
    "\n",
    "office_tuits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>name</th>\n",
       "      <th>percentage</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>name</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Aníbal Ostoa Ortega</th>\n",
       "      <td>13</td>\n",
       "      <td>Aníbal Ostoa Ortega</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Elsa Amabel Landín</th>\n",
       "      <td>4</td>\n",
       "      <td>Elsa Amabel Landín</td>\n",
       "      <td>0.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Lucía Riojas</th>\n",
       "      <td>186</td>\n",
       "      <td>Lucía Riojas</td>\n",
       "      <td>3.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Marcelo Ebrard</th>\n",
       "      <td>2390</td>\n",
       "      <td>Marcelo Ebrard</td>\n",
       "      <td>43.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>María Merced González</th>\n",
       "      <td>1</td>\n",
       "      <td>María Merced González</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Olga Sánchez Cordero</th>\n",
       "      <td>2841</td>\n",
       "      <td>Olga Sánchez Cordero</td>\n",
       "      <td>51.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Víctor Villalobos Arámbula</th>\n",
       "      <td>85</td>\n",
       "      <td>Víctor Villalobos Arámbula</td>\n",
       "      <td>1.5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                             text                         name  percentage\n",
       "name                                                                      \n",
       "Aníbal Ostoa Ortega            13         Aníbal Ostoa Ortega          0.2\n",
       "Elsa Amabel Landín              4           Elsa Amabel Landín         0.1\n",
       "Lucía Riojas                  186                 Lucía Riojas         3.4\n",
       "Marcelo Ebrard               2390               Marcelo Ebrard        43.3\n",
       "María Merced González           1       María Merced González          0.0\n",
       "Olga Sánchez Cordero         2841         Olga Sánchez Cordero        51.5\n",
       "Víctor Villalobos Arámbula     85  Víctor Villalobos Arámbula          1.5"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Transformación de los datos\n",
    "name_tuits = pd.DataFrame({'text' : tuits_labeled.groupby('name')['text'].count()})\n",
    "name_tuits['name'] = name_tuits.index\n",
    "name_tuits['percentage'] = round((name_tuits['text']/sum(name_tuits['text']))*100, 1)\n",
    "\n",
    "# Construcción de la gráfica\n",
    "sequential_colors = sns.color_palette(\"RdPu\", 2)\n",
    "sns.set_palette(sequential_colors)\n",
    "name_plot = sns.barplot(x = 'name', y ='text', data = name_tuits)\n",
    "name_plot.set(xlabel='Persona', ylabel='Número de tuits recopilados', title = \"Distribución de tuits por persona en muestra aleatoria\")\n",
    "\n",
    "name_tuits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>class</th>\n",
       "      <th>percentage</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>class</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>aggresive</th>\n",
       "      <td>1075</td>\n",
       "      <td>aggresive</td>\n",
       "      <td>19.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>neutral</th>\n",
       "      <td>4445</td>\n",
       "      <td>neutral</td>\n",
       "      <td>80.5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           text      class  percentage\n",
       "class                                 \n",
       "aggresive  1075  aggresive        19.5\n",
       "neutral    4445    neutral        80.5"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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K7LnbRpDoLgMmAVc5UCx6XLLwszMcrllYWZmuRrlQIIdOnSI7t2713c1zMw2KdOnT/8gIjYYswsaabDo3r0706ZNq+9qmJltUiS9WWqZL0OZWb269tpr2X333enduzdDhw7ls88+Y+7cuQwYMIBevXpxwgkn8MUXX2yw3rx589hiiy3o27cvffv25fTTTwfg888/55BDDqF3797ceOONa/OPHDmSF17Y5O8XqDcOFmZWb95++21+9atfMW3aNGbNmsXq1au5/fbbOe+88/jZz37GnDlz2HbbbRkzZkzR9Xv27MmMGTOYMWMGN910EwATJ06kX79+vPjii4wePRqAmTNnsmbNGvbaa68627fGxsHCzOrVqlWr+PTTT1m1ahUrVqygU6dOPProowwZMgSA4cOHc++995ZdXsuWLdeWV+XCCy/k0ksvrfW6NyUOFmZWb7p06cI555zDDjvsQKdOnWjTpg39+vWjbdu2tGiRdal27dqVt98uPv7k3Llz2WuvvTjggAN44olsFJGDDjqId999lwEDBnDuuecyfvx4+vXrR+fOdTUIcePUKDu4zWzTsGTJEu677z7mzp1L27ZtOe6443jggQc2yLf+gMOZTp068dZbb9G+fXumT5/OUUcdxcsvv0zr1q354x+zIa5WrlzJoEGDGD9+PGeddRZvvfUWw4YN48gjj9ygPKuZWxZmVm8efvhhevToQceOHWnZsiXHHHMMTz/9NEuXLl17GWnBggVFWwWbbbYZ7dtngx3369ePnj178tprr62X58Ybb2T48OFMnTqVVq1acccddzBq1KjK71gj5GBhZvVmhx124JlnnmHFihVEBI888gi77bYb3/72t7n77rsBGDduHIMHD95g3UWLFrF6dTbq9xtvvMGcOXPYaad1I5AvWbKECRMmMGzYMFasWEGzZs2QxGefbcxPaZiDhZnVmwEDBjBkyBD23ntv+vTpw5o1axg5ciRXXnkl11xzDTvvvDMffvghI0aMAGD8+PFcdNFFAEyZMoU99tiDPffckyFDhnDTTTfRrt26XyS+9NJLueCCC5DEoEGDmDZtGn369OG0006rl33d1DXKsaH69+8ffijPzOzLkTQ9IvoXW+YObrNNzMSB/1XfVbAGaNDUcytavi9DmZlZLgcLMzPL5WBhZma5HCzMzCyXg4WZmeVysDAzs1wOFmZm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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Distribución por agresividad\n",
    "# Transformación de los datos\n",
    "aggresive_tuits = pd.DataFrame({'text' : tuits_labeled.groupby('class')['text'].count()})\n",
    "aggresive_tuits['class'] = aggresive_tuits.index\n",
    "aggresive_tuits['percentage'] = round((aggresive_tuits['text']/sum(aggresive_tuits['text']))*100, 1)\n",
    "\n",
    "\n",
    "# Creación de la gráfica\n",
    "sequential_colors = sns.color_palette(\"RdPu\", 2)\n",
    "sns.set_palette(sequential_colors)\n",
    "aggresive_plot = sns.barplot(x = 'class', y ='text', data = aggresive_tuits)\n",
    "aggresive_plot.set(xlabel='Tipo de tuit', ylabel='Número de tuits', title = \"Distribución de tuits agresivos y neutrales\")\n",
    "\n",
    "aggresive_plot.text(x =-0.1, y = 1150, s = \"19.5%\")\n",
    "aggresive_plot.text(x = 0.9, y = 4480, s = \"80.5%\")\n",
    "\n",
    "\n",
    "aggresive_tuits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>gendered</th>\n",
       "      <th>percentage</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>gendered</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0.0</th>\n",
       "      <td>982</td>\n",
       "      <td>0.0</td>\n",
       "      <td>91.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1.0</th>\n",
       "      <td>93</td>\n",
       "      <td>1.0</td>\n",
       "      <td>8.7</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          text gendered  percentage\n",
       "gendered                           \n",
       "0.0        982      0.0        91.3\n",
       "1.0         93      1.0         8.7"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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347rJC655BKGDRvG0KFDufjiiwH4zW9+w9ChQ+nSpUvFpP3OO++w++67s/POOzN06FDOPntFLenx48ez00478c1vfvODfueddx5Tpxb2nNnMOph18krIVjZr1ix+9rOf8fDDD7Peeutx8MEHc8ghhzBs2DBuueUWvvCFL1Scdv311+fee++lZ8+eLFu2jH322YfRo0fTo0dqVeeJJ55g33335fXXX6exsZGHH36Yb397nXn5qpnVWd2uhCT9QtLLkmaV9Ost6Q+Sns7/N839JelSSXMkPSFpRMk0E/L4T0sq9AdtHdWTTz7JnnvuSY8ePejWrRv77bcft956K4MHD+ZjH6v+Y3pJ9OyZmh5btmwZy5YtQxLdu3dn6dKlvP/++7z77rt07dqV73znO5x77rntsUpmto6o5+24q0nNuJQ6E7gnIgaRmgNvanl3NKkZkEHAJNIrCShpV2wPUhtpZzclLqvdsGHDuP/++3n11VdpbGzkzjvvZN68lhovXuG9995j+PDhbL755owaNYo99tiDwYMHM2DAAEaMGMExxxzDnDlziAh22WWXOq6Jma1r6nY7LiLulzSwWe8jSO+hAbiG9E6NM3L/a3Pz4g/mX/Vumcf9Q0S8BiDpD6TEVmur0AYMHjyYM844g1GjRtGzZ0923nlnunWrfdN37dqVmTNnsmTJEj796U8za9Yshg0b9sGzJYDDDjuMK664gu9973s8/vjjjBo1is9//vP1WB0zW4e0d8WELSJiIUD+v3nu35+V3ysyP/er1H8VkiZJmiFpxqJFi9o88I5u4sSJPPbYY9x///307t2bQYMGtXoevXr1YuTIkfz+9yu3Nzp16lQaGhp4++23mTVrFjfddBPXXXcdjY2NFeZkZpasLbXjVKZfVOm/as+IKyOiISIa+vZt12ruHcLLL78MwPPPP88tt9zCuHHjWpgiWbRoEUuWpDcaLF26lLvvvpsddljxfrBly5ZxySWX8LWvfY3GxkaUX3XU9KzIzKya9k5CL+XbbOT/L+f+81n55VZbkZphr9TfWukzn/kMQ4YM4bDDDuOyyy5j00035dZbb2WrrbZi+vTpHHLIIXzqU+m1MwsWLGDMmDEALFy4kE9+8pPstNNO7LbbbowaNYpDD13xwtTLLruMCRMm0KNHD3baaScigh133JGPf/zj9OrVq5B1NbOOo65t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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Tuits misóginos \n",
    "# Transformación de los datos\n",
    "aggresive = tuits_labeled['class'] == \"aggresive\"\n",
    "aggresive_tuits = tuits_labeled[aggresive]\n",
    "\n",
    "misogynistic_tuits = pd.DataFrame({'text' : aggresive_tuits.groupby('gendered')['text'].count()})\n",
    "misogynistic_tuits['gendered'] = misogynistic_tuits.index\n",
    "misogynistic_tuits['percentage'] = round((misogynistic_tuits['text']/sum(misogynistic_tuits['text']))*100, 1)\n",
    "\n",
    "\n",
    "# Creación de la gráfica\n",
    "sequential_colors = sns.color_palette(\"RdPu\", 2)\n",
    "sns.set_palette(sequential_colors)\n",
    "misogynistic_plot = sns.barplot(x = 'gendered', y ='text', data = misogynistic_tuits)\n",
    "misogynistic_plot.set(xlabel='Tipo de tuit', ylabel='Número de tuits', title = \"Distribución de tuits agresivos entre misóginos y no misóginos\")\n",
    "\n",
    "misogynistic_plot.text(x =-0.1, y = 990, s = \"91.3%\")\n",
    "misogynistic_plot.text(x = 0.9, y = 120, s = \"8.7%\")\n",
    "\n",
    "\n",
    "misogynistic_tuits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Separamos las características a predecir\n",
    "x_all = tuits_labeled.drop(['class', 'gendered'], axis = 1)\n",
    "y_all = tuits_labeled[['class', 'gendered']]\n",
    "num_test= 0.8\n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size = 0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(4416, 4)\n",
      "(4416, 2)\n",
      "(1104, 4)\n",
      "(1104, 2)\n"
     ]
    }
   ],
   "source": [
    "# Vemos las dimensiones de las bases de entrenamiento y de prueba\n",
    "print(x_train.shape)\n",
    "print(y_train.shape)\n",
    "print(x_test.shape)\n",
    "print(y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.series.Series'>\n",
      "\n",
      "\n",
      "Estado original text        object\n",
      "name      category\n",
      "gender    category\n",
      "office    category\n",
      "dtype: object\n"
     ]
    }
   ],
   "source": [
    "print(type(x_train['text']))\n",
    "print(\"\\n\")\n",
    "print(\"Estado original\", x_train.dtypes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "ename": "MemoryError",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mMemoryError\u001b[0m                               Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-90-1b118b35a129>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# Intentos que pierden la estructura bidimensional y pasan a unidimensional\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0ma\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mstr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Intento 1\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mx_train\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdtypes\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"\\n\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\pandas\\core\\series.py\u001b[0m in \u001b[0;36m__repr__\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m   1369\u001b[0m             \u001b[0mmin_rows\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmin_rows\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1370\u001b[0m             \u001b[0mmax_rows\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmax_rows\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1371\u001b[1;33m             \u001b[0mlength\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mshow_dimensions\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1372\u001b[0m         )\n\u001b[0;32m   1373\u001b[0m         \u001b[0mresult\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mbuf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mgetvalue\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\pandas\\core\\series.py\u001b[0m in \u001b[0;36mto_string\u001b[1;34m(self, buf, na_rep, float_format, header, index, length, dtype, name, max_rows, min_rows)\u001b[0m\n\u001b[0;32m   1435\u001b[0m             \u001b[0mmax_rows\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmax_rows\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1436\u001b[0m         )\n\u001b[1;32m-> 1437\u001b[1;33m         \u001b[0mresult\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mformatter\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mto_string\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1438\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1439\u001b[0m         \u001b[1;31m# catch contract violations\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\pandas\\io\\formats\\format.py\u001b[0m in \u001b[0;36mto_string\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    385\u001b[0m             \u001b[0mresult\u001b[0m \u001b[1;33m+=\u001b[0m \u001b[1;34m\"\\n\"\u001b[0m \u001b[1;33m+\u001b[0m \u001b[0mfooter\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    386\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 387\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mstr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"\"\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mjoin\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mresult\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    388\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    389\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mMemoryError\u001b[0m: "
     ]
    }
   ],
   "source": [
    "# Intentos que pierden la estructura bidimensional y pasan a unidimensional\n",
    "a = str(x_train.iloc[:, 0])\n",
    "print(\"Intento 1\", x_train.dtypes)\n",
    "print(\"\\n\")\n",
    "\n",
    "b = (x_train['text'])\n",
    "print(\"Intento 2\", x_train.dtypes)\n",
    "print(\"\\n\")\n",
    "\n",
    "\n",
    "# Intentos que no cambian el tipo de dato \n",
    "x_train['text'].astype('str')\n",
    "print(\"Intento 3\", x_train.dtypes)\n",
    "print(\"\\n\")\n",
    "\n",
    "x_train.text.astype(str)\n",
    "print(\"Intento 4\", x_train.dtypes)\n",
    "print(\"\\n\")\n",
    "\n",
    "x_train.iloc[:, 0] = x_train.text.astype(str)\n",
    "print(\"Intento 5\", x_train.dtypes)\n",
    "print(\"\\n\")\n",
    "\n",
    "# Intentos que ni siquiera corren\n",
    "x_train['text'] = x_train.text.astype(str)\n",
    "print(\"Intento 6\", x_train.dtypes)\n",
    "\n",
    "x_train['text'] = str(x_train.iloc[:, 0])\n",
    "print(\"Intento 7\", x_train.dtypes)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Miniconda3\\lib\\site-packages\\ipykernel_launcher.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  \"\"\"Entry point for launching an IPython kernel.\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "module 'pandas' has no attribute 'to_numpy'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-47-d12c53130e55>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mto_numpy\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx_test\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\pandas\\__init__.py\u001b[0m in \u001b[0;36m__getattr__\u001b[1;34m(name)\u001b[0m\n\u001b[0;32m    261\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0m_SparseArray\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    262\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 263\u001b[1;33m         \u001b[1;32mraise\u001b[0m \u001b[0mAttributeError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34mf\"module 'pandas' has no attribute '{name}'\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    264\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    265\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mAttributeError\u001b[0m: module 'pandas' has no attribute 'to_numpy'"
     ]
    }
   ],
   "source": [
    "pd.to_numpy(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cambiar el tipo de dato del texto \n",
    "#x_train = print(x_train.to_string())\n",
    "x_train = str(x_train)\n",
    "x_test = str(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "#x_test = print(x_test.to_string())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "text_train = str(x_train['text'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "'NoneType' object is not subscriptable",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-28-da65d7cd298a>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mtype\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'text'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m: 'NoneType' object is not subscriptable"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Preprocessor is $HASHTAG$ $EMOJI$ $URL$\n",
      "Preprocessor is\n",
      "<preprocessor.parse.ParseResult object at 0x000002C1A4193E88>\n"
     ]
    }
   ],
   "source": [
    "# Procesamiento de texto con Tweet-Preprocessor\n",
    "# Ejemplo de uso \n",
    "print(p.tokenize('Preprocessor is #awesome 👍 https://github.com/s/preprocessor'))\n",
    "print(p.clean('Preprocessor is #awesome 👍 https://github.com/s/preprocessor'))\n",
    "print(p.parse('Preprocessor is #awesome https://github.com/s/preprocessor'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "expected string or bytes-like object",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-64-8ff74f5cc59e>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtokenize\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx_test\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtokenize\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\preprocessor\\api.py\u001b[0m in \u001b[0;36mtokenize\u001b[1;34m(tweet_string)\u001b[0m\n\u001b[0;32m     41\u001b[0m         \u001b[0mPreprocessor\u001b[0m \u001b[1;32mis\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m$\u001b[0m\u001b[0mHASHTAG\u001b[0m\u001b[0;31m$\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m$\u001b[0m\u001b[0mURL\u001b[0m\u001b[0;31m$\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     42\u001b[0m     \"\"\"\n\u001b[1;32m---> 43\u001b[1;33m     \u001b[0mtokenized_tweet_string\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpreprocessor\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mclean\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtweet_string\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mFunctions\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mTOKENIZE\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     44\u001b[0m     \u001b[1;32mreturn\u001b[0m \u001b[0mtokenized_tweet_string\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     45\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\preprocessor\\preprocess.py\u001b[0m in \u001b[0;36mclean\u001b[1;34m(self, tweet_string, repl)\u001b[0m\n\u001b[0;32m     29\u001b[0m                 \u001b[0mtweet_string\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmethod_to_call\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtweet_string\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m''\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     30\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 31\u001b[1;33m                 \u001b[0mtweet_string\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmethod_to_call\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtweet_string\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtoken\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     32\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     33\u001b[0m         \u001b[0mtweet_string\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mremove_unneccessary_characters\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtweet_string\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\preprocessor\\preprocess.py\u001b[0m in \u001b[0;36mpreprocess_emojis\u001b[1;34m(self, tweet_string, repl)\u001b[0m\n\u001b[0;32m     47\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     48\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mpreprocess_emojis\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtweet_string\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrepl\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 49\u001b[1;33m         \u001b[0mprocessed\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mPatterns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mEMOJIS_PATTERN\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msub\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mrepl\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtweet_string\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     50\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0mprocessed\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mencode\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'ascii'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'ignore'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdecode\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'ascii'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     51\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: expected string or bytes-like object"
     ]
    }
   ],
   "source": [
    "p.tokenize(x_test)\n",
    "p.tokenize(x_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'str'>\n"
     ]
    },
    {
     "ename": "TypeError",
     "evalue": "expected string or bytes-like object",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-55-beae47cd89d6>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     12\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     13\u001b[0m \u001b[1;31m# Aplicar las transformaciones\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 14\u001b[1;33m \u001b[0mx_train_processed\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtextmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtransform\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtextmodel\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\microtc\\textmodel.py\u001b[0m in \u001b[0;36mtransform\u001b[1;34m(self, texts)\u001b[0m\n\u001b[0;32m    374\u001b[0m         \u001b[1;33m>>\u001b[0m\u001b[1;33m>\u001b[0m \u001b[0mX\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtextmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtransform\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcorpus\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    375\u001b[0m         \"\"\"\n\u001b[1;32m--> 376\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtonp\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__getitem__\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mx\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mtexts\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    377\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    378\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mvectorize\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtext\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\microtc\\textmodel.py\u001b[0m in \u001b[0;36m<listcomp>\u001b[1;34m(.0)\u001b[0m\n\u001b[0;32m    374\u001b[0m         \u001b[1;33m>>\u001b[0m\u001b[1;33m>\u001b[0m \u001b[0mX\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtextmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtransform\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcorpus\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    375\u001b[0m         \"\"\"\n\u001b[1;32m--> 376\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtonp\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__getitem__\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mx\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mtexts\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    377\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    378\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mvectorize\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtext\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\microtc\\textmodel.py\u001b[0m in \u001b[0;36m__getitem__\u001b[1;34m(self, text)\u001b[0m\n\u001b[0;32m    342\u001b[0m         \u001b[1;33m:\u001b[0m\u001b[0mrtype\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mlist\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    343\u001b[0m         \"\"\"\n\u001b[1;32m--> 344\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mmodel\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtokenize\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtext\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    345\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    346\u001b[0m     \u001b[1;33m@\u001b[0m\u001b[0mclassmethod\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\microtc\\textmodel.py\u001b[0m in \u001b[0;36mtokenize\u001b[1;34m(self, text)\u001b[0m\n\u001b[0;32m    409\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mtokens\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    410\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 411\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_tokenize\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtext\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    412\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    413\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mget_text\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtext\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\microtc\\textmodel.py\u001b[0m in \u001b[0;36m_tokenize\u001b[1;34m(self, text)\u001b[0m\n\u001b[0;32m    543\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    544\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_tokenize\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtext\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 545\u001b[1;33m         \u001b[0mtext\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtext_transformations\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtext\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    546\u001b[0m         \u001b[0mL\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    547\u001b[0m         \u001b[1;32mfor\u001b[0m \u001b[0m_\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcompute_tokens\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtext\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\microtc\\textmodel.py\u001b[0m in \u001b[0;36mtext_transformations\u001b[1;34m(self, text)\u001b[0m\n\u001b[0;32m    446\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    447\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0memo_map\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 448\u001b[1;33m             \u001b[0mtext\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0memo_map\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mreplace\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtext\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0moption\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0memo_option\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    449\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    450\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mselect_ent\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\microtc\\emoticons.py\u001b[0m in \u001b[0;36mreplace\u001b[1;34m(self, text, option)\u001b[0m\n\u001b[0;32m     84\u001b[0m                 \u001b[0mklass\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m''\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     85\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 86\u001b[1;33m             \u001b[0mtext\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpat\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msub\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mklass\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtext\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     87\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     88\u001b[0m         \u001b[0mT\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: expected string or bytes-like object"
     ]
    }
   ],
   "source": [
    "# Procesamiento de texto con TextModel\n",
    "\n",
    "#Importar la paquetería \n",
    "from microtc.textmodel import TextModel\n",
    "\n",
    "# Establecer el objeto que contiene el texto que será procesado\n",
    "corpus = text_train\n",
    "print(type(corpus))\n",
    "\n",
    "# Ajustar a un modelo de texto\n",
    "textmodel = TextModel().fit(corpus)\n",
    "\n",
    "# Aplicar las transformaciones\n",
    "x_train_processed = textmodel.transform(textmodel)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,\n",
       "                       criterion='entropy', max_depth=5, max_features='sqrt',\n",
       "                       max_leaf_nodes=None, max_samples=None,\n",
       "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "                       min_samples_leaf=1, min_samples_split=2,\n",
       "                       min_weight_fraction_leaf=0.0, n_estimators=20,\n",
       "                       n_jobs=None, oob_score=False, random_state=None,\n",
       "                       verbose=0, warm_start=False)"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "# Seleccionamos el tipo de clasificador, en este caso RandomForest. \n",
    "clf = RandomForestClassifier(n_estimators=20, max_features='sqrt', criterion='entropy', max_depth=5)\n",
    "clf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,\n",
       "                       criterion='gini', max_depth=None, max_features='auto',\n",
       "                       max_leaf_nodes=None, max_samples=None,\n",
       "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "                       min_samples_leaf=1, min_samples_split=2,\n",
       "                       min_weight_fraction_leaf=0.0, n_estimators=100,\n",
       "                       n_jobs=None, oob_score=False, random_state=None,\n",
       "                       verbose=0, warm_start=False)"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf2 = RandomForestClassifier()\n",
    "clf2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "could not convert string to float: '                                                                                                                                                text  \\\\\\n3181    RT @ruloprado: @Yelysgatub3la @PaRiS_CGxM @luis_journalist @m_ebrard @julioastillero @SGarciaSoto @lumendoz @lopezdoriga @AristeguiOnline @\\x85   \\n5344    Y quién es ese contador? Tendría que ir a la cárcel, él y sus \"beneficiados\"pues según @lopezobrador_ \"nadie por en\\x85 https://t.co/LQk4qRzApg   \\n1998    RT @LeonKrauze: Secretaria @M_OlgaSCordero , con todo respeto: ¿esto es lo que decide usted compartir esta noche? ¿De verdad? https://t.co/\\x85   \\n3163     @SEGOB_mx @GobiernoMX @M_OlgaSCordero @lopezobrador_ @SSPCMexico ES MUY CONDENABLE, INADMISIBLE, INSOPORTABLE, MUY\\x85 https://t.co/LqHsR2Nt8s   \\n3659                                                                  @m_ebrard carta de condolencias a Sara sosa! Perfecto! Pero y los otros hijos?   \\n...                                                                                                                                              ...   \\n3206                         RT @DiazBimba: Por favor detengan a la Wallace!!! @lopezobrador_ @AlfonsoDurazo @M_OlgaSCordero https://t.co/d48cuTS5LM   \\n1782  RT @ajplusespanol: Y al final, los legisladores gritaban \\x93¡que lo prenda, que lo prenda!\\x94:\\\\n\\\\nLa secretaria de Gobernación @M_OlgaSCordero te\\x85   \\n1177    @Guardiolal @gudcom @M_OlgaSCordero Osea que ya debemos estar acostumbrados ya no debe ser noticia o será  hasta qu\\x85 https://t.co/OKggMgrbiZ   \\n1289    @LuRiojas Estás encontrar de la violencia hacia la mujer, verdad?!! Bueno, veo q llevas un pañuelo verde, (abortist\\x85 https://t.co/QdqIiNFeWi   \\n2679     RT @JoseLuisBeato: Excelente reunión con el Canciller @m_ebrard titular de @SRE_mx junto a los Secretarios integrantes de la @amsdemex. En\\x85   \\n\\n                      name  gender     office  \\n3181        Marcelo Ebrard    male  executive  \\n5344  Olga Sánchez Cordero  female  executive  \\n1998  Olga Sánchez Cordero  female  executive  \\n3163  Olga Sánchez Cordero  female  executive  \\n3659        Marcelo Ebrard    male  executive  \\n...                    ...     ...        ...  \\n3206  Olga Sánchez Cordero  female  executive  \\n1782  Olga Sánchez Cordero  female  executive  \\n1177  Olga Sánchez Cordero  female  executive  \\n1289          Lucía Riojas  female   congress  \\n2679        Marcelo Ebrard    male  executive  \\n\\n[4416 rows x 4 columns]'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-40-3a52b89209d6>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# Entrenamiento del algoritmo\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mclf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\sklearn\\ensemble\\_forest.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[0;32m    293\u001b[0m         \"\"\"\n\u001b[0;32m    294\u001b[0m         \u001b[1;31m# Validate or convert input data\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 295\u001b[1;33m         \u001b[0mX\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maccept_sparse\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m\"csc\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mDTYPE\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    296\u001b[0m         \u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maccept_sparse\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'csc'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mensure_2d\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    297\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0msample_weight\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\sklearn\\utils\\validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[1;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)\u001b[0m\n\u001b[0;32m    529\u001b[0m                     \u001b[0marray\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0marray\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mastype\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcasting\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m\"unsafe\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    530\u001b[0m                 \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 531\u001b[1;33m                     \u001b[0marray\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0marray\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0morder\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0morder\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    532\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mComplexWarning\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    533\u001b[0m                 raise ValueError(\"Complex data not supported\\n\"\n",
      "\u001b[1;32mC:\\ProgramData\\Miniconda3\\lib\\site-packages\\numpy\\core\\_asarray.py\u001b[0m in \u001b[0;36masarray\u001b[1;34m(a, dtype, order)\u001b[0m\n\u001b[0;32m     83\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     84\u001b[0m     \"\"\"\n\u001b[1;32m---> 85\u001b[1;33m     \u001b[1;32mreturn\u001b[0m \u001b[0marray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0ma\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0morder\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0morder\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     86\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     87\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: could not convert string to float: '                                                                                                                                                text  \\\\\\n3181    RT @ruloprado: @Yelysgatub3la @PaRiS_CGxM @luis_journalist @m_ebrard @julioastillero @SGarciaSoto @lumendoz @lopezdoriga @AristeguiOnline @\\x85   \\n5344    Y quién es ese contador? Tendría que ir a la cárcel, él y sus \"beneficiados\"pues según @lopezobrador_ \"nadie por en\\x85 https://t.co/LQk4qRzApg   \\n1998    RT @LeonKrauze: Secretaria @M_OlgaSCordero , con todo respeto: ¿esto es lo que decide usted compartir esta noche? ¿De verdad? https://t.co/\\x85   \\n3163     @SEGOB_mx @GobiernoMX @M_OlgaSCordero @lopezobrador_ @SSPCMexico ES MUY CONDENABLE, INADMISIBLE, INSOPORTABLE, MUY\\x85 https://t.co/LqHsR2Nt8s   \\n3659                                                                  @m_ebrard carta de condolencias a Sara sosa! Perfecto! Pero y los otros hijos?   \\n...                                                                                                                                              ...   \\n3206                         RT @DiazBimba: Por favor detengan a la Wallace!!! @lopezobrador_ @AlfonsoDurazo @M_OlgaSCordero https://t.co/d48cuTS5LM   \\n1782  RT @ajplusespanol: Y al final, los legisladores gritaban \\x93¡que lo prenda, que lo prenda!\\x94:\\\\n\\\\nLa secretaria de Gobernación @M_OlgaSCordero te\\x85   \\n1177    @Guardiolal @gudcom @M_OlgaSCordero Osea que ya debemos estar acostumbrados ya no debe ser noticia o será  hasta qu\\x85 https://t.co/OKggMgrbiZ   \\n1289    @LuRiojas Estás encontrar de la violencia hacia la mujer, verdad?!! Bueno, veo q llevas un pañuelo verde, (abortist\\x85 https://t.co/QdqIiNFeWi   \\n2679     RT @JoseLuisBeato: Excelente reunión con el Canciller @m_ebrard titular de @SRE_mx junto a los Secretarios integrantes de la @amsdemex. En\\x85   \\n\\n                      name  gender     office  \\n3181        Marcelo Ebrard    male  executive  \\n5344  Olga Sánchez Cordero  female  executive  \\n1998  Olga Sánchez Cordero  female  executive  \\n3163  Olga Sánchez Cordero  female  executive  \\n3659        Marcelo Ebrard    male  executive  \\n...                    ...     ...        ...  \\n3206  Olga Sánchez Cordero  female  executive  \\n1782  Olga Sánchez Cordero  female  executive  \\n1177  Olga Sánchez Cordero  female  executive  \\n1289          Lucía Riojas  female   congress  \\n2679        Marcelo Ebrard    male  executive  \\n\\n[4416 rows x 4 columns]'"
     ]
    }
   ],
   "source": [
    "# Entrenamiento del algoritmo \n",
    "clf.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Se hacen las predicciones sobre el test\n",
    "from sklearn.metrics import make_scorer, accuracy_score\n",
    "predictions = clf.predict(X_test)\n",
    "# se imprime el accuracy\n",
    "print(accuracy_score(y_test, predictions))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

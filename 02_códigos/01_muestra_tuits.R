#------------------------------------------------------------------------------#
# Proyecto:                   Tesis: Seximo en código binario 
# Objetivo:                   Generar datos descriptivos de la base de tuits 
#
# Encargada:                  Regina Isabel Medina Rosales
# Correo:                     regina.medina@alumnos.cide.edu
# Fecha de creación:          01 de marzo de 2021
# Última actualización:       03 de marzo de 2021
#------------------------------------------------------------------------------#

# 00. Configuración inicial ----------------------------------------------------
# Cargar librerías 
library(pacman)
p_load(readr, dplyr, extrafont, ggplot2)


# Limpiar espacio de trabajo 
rm(list=ls())

# Configurar directorios
inp_data <- "Tuits/"
out_figs <- "Figuras/"


# 01. Cargar datos -------------------------------------------------------------
df_crudo <- read_csv(paste0(inp_data, "/tuits_labeled_utf8.csv"))

# 02. Limpiar datos ------------------------------------------------------------
# Traducir valores y seleccionar variables finales
df_limpio_esp <- df_crudo %>% 
    # Corregir acentos de los nombres propios
    mutate(nombre = case_when(name == "An?bal Ostoa Ortega" ~ "Aníbal Ostoa Ortega", 
                            name == "Elsa Amabel Land?n"    ~ "Elsa Amabel Landín",
                            name == "Luc?a Riojas"          ~ "Lucía Riojas",
                            name == "Mar?a Merced Gonz?lez" ~ "María Merced González",
                            name == "Marcelo Ebrard"        ~ "Marcelo Ebrard",
                            name == "Olga S?nchez Cordero"  ~ "Olga Sánchez Cordero",
                            name == "V?ctor Villalobos Ar?mbula" ~ "V?ctor Villalobos Arámbula")) %>% 
    # Traducir sexo
    mutate(sexo = case_when(gender == "female"       ~ "Mujer",
                            gender == "male"         ~ "Hombre")) %>% 
    # Traducir cargo   
    mutate(cargo = case_when(office == "congress"    ~ "Congreso federal",
                            office == "executive"    ~ "Gabinete federal",
                            office == "local"        ~ "Congreso local",
                            office == "senate"       ~ "Senado (federal)")) %>% 
    # Traducir indicador binario de violencia
    mutate(violento = case_when(class == "aggresive" ~ "Violento",
                            class == "neutral"       ~ "Neutral")) %>% 
    # Traducir indicador binario de violencia
    mutate(sexista = case_when(gendered == 0         ~ "No sexista", 
                                gendered == 1        ~ "Sexista")) %>% 
    # Seleccionar variables finales
    select(id, user, text, nombre, sexo, cargo, violento, sexista)

# Revisar valores 
table(df_limpio_esp$nombre)
table(df_limpio_esp$sexo)
table(df_limpio_esp$cargo)
table(df_limpio_esp$violento)
table(df_limpio_esp$sexista)

# Agrupar por categorías para entender distribución de la submuestra
# Por tipo de cargo 
df_cargo <- df_limpio_esp                   %>% 
    group_by(cargo)                         %>% 
    summarise(total = n())                  %>% 
    mutate(porcentaje = round(total*100/sum(total), 1), 
            por_text = paste0(porcentaje, "%"), 
        num_por = paste0(por_text, "\n", "(n = ", total, ")"))
  
# Por sexo   
df_sexo <- df_limpio_esp                    %>%  
    group_by(sexo)                          %>% 
    summarise(total = n())                  %>% 
    mutate(porcentaje = round(total*100/sum(total), 1), 
        por_text = paste0(porcentaje, "%"), 
        num_por = paste0(por_text, "\n", "(n = ", total, ")"))  
        
# Por indicador de violencia
df_violento <- df_limpio_esp                %>%     
    group_by(violento)                      %>% 
    summarise(total = n())                  %>% 
    mutate(porcentaje = round(total*100/sum(total), 1), 
        por_text = paste0(porcentaje, "%"), 
        num_por = paste0(por_text, "\n", "(n = ", total, ")")) 

# Por indicador de violencia y sexp
df_violento_sexo <- df_limpio_esp           %>% 
    group_by(violento, sexo)                %>% 
    summarise(total = n())                  %>%
    group_by(sexo)                          %>% 
    mutate(porcentaje = round(total*100/sum(total), 1), 
        por_text = paste0(porcentaje, "%"), 
        num_por = paste0(por_text, "\n", "(n = ", total, ")")) 

# Por indicador de misoginia
df_sexista <- df_limpio_esp                 %>% 
    group_by(sexista)                       %>% 
    summarise(total = n())                  %>% 
    mutate(porcentaje = round(total*100/sum(total), 1), 
        por_text = paste0(porcentaje, "%"), 
        num_por = paste0(por_text, "\n", "(n = ", total, ")")) 

# Por indicador de misoginia sólo en tuits violentos
df_violento_sexista <- df_limpio_esp        %>% 
    filter(violento == "Violento")          %>% 
    group_by(sexista)                       %>% 
    summarise(total = n())      %>% 
    mutate(porcentaje = round(total*100/sum(total), 1), 
        por_text = paste0(porcentaje, "%"), 
        num_por = paste0(por_text, "\n", "(n = ", total, ")")) 

# Por indicador de misoginia sólo en tuits violentos contra mujeres
df_violento_sexista_sexo <- df_limpio_esp           %>% 
    filter(violento == "Violento")                  %>% 
    group_by(sexista, sexo)                         %>% 
    summarise(total = n())                          %>%
    group_by(sexo)                                  %>% 
    mutate(porcentaje = round(total*100/sum(total), 1), 
        por_text = paste0(porcentaje, "%"), 
        num_por = paste0(por_text, "\n", "(n = ", total, ")")) 

# 03. Gráficas -----------------------------------------------------------------
# 03.1 Establecer vectores -----------------------------------------------------
# Etiquetas 
lab_y <- "Porcentaje (%)\n"

# Colores 
v_colors_sex <- c("#355070", "#B56576")
v_colors_viol <- c("#EAAC8B", "#6D597A")

# Formato de imagen 
v_format <- ".png"

# Fuente tipográfica 
    # fonts()                        # Consultar fuentes disponibles
v_fuente <- "Arial"
choose_font(c(v_fuente))    # Escoger fuente

# Asignar tipografía al tema 
tema <- theme_bw() +
    theme(text        = element_text(family = v_fuente), 
        plot.title    = element_text(family = v_fuente), 
        plot.subtitle = element_text(family = v_fuente), 
        plot.caption  = element_text(family = v_fuente), 
        legend.title  = element_text(family = v_fuente), 
        legend.text   = element_text(family = v_fuente), 
        axis.title    = element_text(family = v_fuente), 
        axis.text.y   = element_text(family = v_fuente), 
        axis.text.x   = element_text(family = v_fuente), 
        strip.text.y  = element_text(family = v_fuente), 
        strip.text.x  = element_text(family = v_fuente))


# 03.2 Generar visualizaciones -------------------------------------------------
# Distribución por cargo 
ggplot(df_cargo, aes(x = cargo, y = porcentaje, fill = "#EAAC8B")) +
    geom_col() +
    geom_text(aes(label = num_por), nudge_y = 6, size = 3, family = v_fuente) +
    labs(title = "Distribución de la submuestra", 
        subtitle = "Según el cargo de la figura pública\n", 
        y = lab_y, 
        x = "") +
    scale_fill_manual(values = v_colors_viol[1]) +
    guides(fill = "none") +
    tema

ggsave(paste0(out_figs, "g_dist_cargo", v_format), width = 7, height = 5)

# Distribución por género
ggplot(df_sexo, aes(x = sexo, y = porcentaje, fill = sexo)) +
    geom_col() +
    geom_text(aes(label = num_por), nudge_y = 4, size = 3, family = v_fuente) +
    labs(title = "Distribución de la submuestra", 
        subtitle = " Según el género de la figura pública\n", 
        y = lab_y, 
        x = "") +
    scale_fill_manual(values = v_colors_sex) +
    guides(fill = "none") +
    tema 

ggsave(paste0(out_figs, "g_dist_genero", v_format), width = 7, height = 5)

# Distribución por tipo violento
ggplot(df_violento, aes(x = violento, y = porcentaje, fill = "#6D597A")) +
    geom_col(alpha = c(0.8, 1)) +
    geom_text(aes(label = num_por), nudge_y = 6, size = 3, family = v_fuente) +
    labs(title = "Distribución de la submuestra según el tipo de comentario", 
        subtitle = "Entre neutral y violento\n", 
        y = lab_y, 
        x = "") +
    scale_fill_manual(values = v_colors_viol[2]) +
    guides(fill = "none") +
    tema

ggsave(paste0(out_figs, "g_violento", v_format), width = 7, height = 5)


# Distribución tipo violento por género 
ggplot(df_violento_sexo, aes(x = violento, y = porcentaje, fill = sexo)) +
    geom_col(alpha = c(0.8, 1, 0.8, 1)) +
    geom_text(aes(label = num_por), nudge_y = 6, size = 3, family = v_fuente) +
    labs(title = "Distribución de la submuestra según el tipo de comentario", 
        subtitle = "Entre neutral y violento; desagregado por género de la figura pública\n", 
        y = lab_y, 
        x = "") +
    facet_wrap(~sexo) +
    scale_fill_manual(values = v_colors_sex) +
    guides(fill = "none") +
    tema 

ggsave(paste0(out_figs, "g_violento_genero", v_format), width = 7, height = 5)


# Distribución tipo sexista de tuits violentos
ggplot(df_violento_sexista, aes(x = sexista, y = porcentaje, fill = "#6D597A")) +
    geom_col(alpha = c(0.8, 1)) +
    geom_text(aes(label = num_por), nudge_y = 6, size = 3, family = v_fuente) +
    labs(title = "Comentarios violentos según la carga de género", 
        subtitle = "Entre sexista y no sexista\n", 
        y = lab_y, 
        x = "") +
    scale_fill_manual(values = v_colors_viol[2]) +
    guides(fill = "none") +
    tema

ggsave(paste0(out_figs, "g_sexista", v_format), width = 7, height = 5)


# Distribución tipo sexista de tuits violentos por género
ggplot(df_violento_sexista_sexo, aes(x = sexista, y = porcentaje, fill = sexo)) +
    geom_bar(stat = "identity", alpha = c(0.8, 1, 0.8, 1)) +
    geom_text(aes(label = num_por), nudge_y = 6, size = 3, family = v_fuente) +
    labs(title = "Comentarios violentos según la carga de género", 
        subtitle = "Entre sexista y no sexista; desagregado por género de la figura pública\n", 
        y = lab_y, 
        x = "") +
    facet_wrap(~sexo) +
    scale_fill_manual(values = v_colors_sex) +
    guides(fill = "none") +
    tema

ggsave(paste0(out_figs, "g_sexista_genero", v_format), width = 7, height = 5)

# FIN --------------------------------------------------------------------------
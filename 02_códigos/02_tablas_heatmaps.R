#------------------------------------------------------------------------------#
# Proyecto:                   Tesis: Seximo en código binario 
# Objetivo:                   Generar tablas de calor de códigos
#
# Encargada:                  Regina Isabel Medina Rosales
# Correo:                     regina.medina@alumnos.cide.edu
# Fecha de creación:          21 de abril de 2021
# Última actualización:       03 de mayo  de 2021
#------------------------------------------------------------------------------#


# 00. Configuración inicial ----------------------------------------------------
# Cargar librerías 
library(pacman)
p_load(readxl, dplyr, stringr, tidyr, ggplot2)


# Limpiar espacio de trabajo 
rm(list=ls())

# Configurar directorios
inp_data <- "Entrevistas/Tablas_análisis/"
out_figs <- "Figuras/"


# 01. Cargar datos -------------------------------------------------------------

df_crudo_efect  <- read_excel(paste0(inp_data, "/Co-ocurrencia-violencias-efectos.xlsx"))
df_crudo_react  <- read_excel(paste0(inp_data, "/Co-ocurrencia-violencias-reacción.xlsx"))

# 02. Limpiar datos ------------------------------------------------------------

# 02.1. Tabla de los efectos de las agresiones ---------------------------------
# Nombres de las variables en la 
v_names     <- names(df_crudo_efect)

# Limpiar nombres y variables
df_efecto  <- df_crudo_efect                %>% 
    rename(
        efecto   = v_names[1], 
        genero   = v_names[2], 
        digital  = v_names[3], 
        politica = v_names[4])              %>% 
    mutate(# total = as.numeric(str_extract(efecto, "[[:digit:]]+")), # Poner frecuencia (números) en otra variable
        efecto = str_extract(efecto, "[[\\w\\s]]+"),    # Quitar números y caracteres especiales 
        efecto = str_sub(efecto, 2, -4))    %>% # Quitar "Gr" y espacios en blanco 
    pivot_longer(cols = genero:politica, 
        names_to = c("violencia"), 
        values_to = "total")                %>% 
    group_by(violencia)                     %>% 
    mutate(porcent = round(total*100/sum(total), 1), 
            text = paste0(porcent, "%\n(n = ", total, ")")) %>% 
    # Mejorar el texto 
    mutate(violencia = case_when(violencia == "digital"   ~ "Violencia digital", 
                                 violencia == "genero"    ~ "Violencia de género", 
                                 violencia == "politica"  ~ "Violencia política"))


# 02.2. Tabla de las reacciones a las agresiones -------------------------------
# Nombres de las variables en la 
v_names     <- names(df_crudo_react)

# Limpiar nombres y variables
df_reaccion  <- df_crudo_react                          %>% 
    rename(
        reaccion = v_names[1], 
        genero   = v_names[2], 
        digital  = v_names[3], 
        politica = v_names[4])                          %>% 
    mutate(# total = as.numeric(str_extract(efecto, "[[:digit:]]+")), # Poner frecuencia (números) en otra variable
        reaccion = str_extract(reaccion, "[[\\w\\s]]+"),    # Quitar números y caracteres especiales 
        reaccion = str_sub(reaccion, 2, -4))            %>% # Quitar "Gr" y espacios en blanco 
    pivot_longer(cols = genero:politica, 
        names_to  = c("violencia"), 
        values_to = "total")                            %>% 
    # Eliminar categoría repetida 
    filter(reaccion != "Cambio en comportamiento\r")    %>% 
    group_by(violencia)                                 %>% 
    mutate(porcent = round(total*100/sum(total), 1), 
        text = paste0(porcent, "%\n(n = ", total, ")")) %>% 
    # Mejorar el texto 
    mutate(violencia = case_when(violencia == "digital"  ~ "Violencia digital", 
        violencia == "genero"  ~ "Violencia de género", 
        violencia == "politica"  ~ "Violencia política")) 



# 03. Visualizaciones ----------------------------------------------------------
# 03.1 Establecer vectores -----------------------------------------------------

# Formato de imagen 
v_format <- ".png"

# Fuente tipográfica 
# fonts()                        # Consultar fuentes disponibles
v_fuente <- "Arial"
choose_font(v_fuente)    # Escoger fuente


# Asignar fuente al tema 
tema <- theme_minimal() +
    theme(
        # Cambiar tipografía a la fuente deseada
        text        = element_text(family = v_fuente), 
        plot.title    = element_text(family = v_fuente, size = 20), 
        plot.subtitle = element_text(family = v_fuente, size = 16), 
        plot.caption  = element_text(family = v_fuente), 
        legend.title  = element_text(family = v_fuente), 
        legend.text   = element_text(family = v_fuente), 
        axis.title    = element_text(family = v_fuente, size = 14), 
        axis.text.y   = element_text(family = v_fuente), 
        axis.text.x   = element_text(family = v_fuente), 
        strip.text.y  = element_text(family = v_fuente), 
        strip.text.x  = element_text(family = v_fuente),
        # Camibar el fondo del panel 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(fill=NA,color="gray90", size=0.5, linetype="solid"),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        panel.background = element_rect(fill="gray90"),
        plot.background = element_rect(fill="gray90"),
        # legend.position = "none", 
        axis.text = element_text(color="black", size=14) 
        )


# 03.2 Generar visualizaciones -------------------------------------------------
# Tabla de los efectos de las agresiones 

ggplot(df_efecto, aes(x = violencia, y = efecto, fill = porcent)) +
    geom_tile(color = "gray90", size = 1.5) +
    geom_text(aes(label = text), family = v_fuente, color = "black", 
        size = rel(4)) +
    labs(title = "Tipos de efectos de las agresiones", 
        subtitle = "Por dimensión de violencia", 
        x = "", 
        y = "", 
        fill = "Frecuencia") +
    scale_x_discrete(position = "top") +
    scale_y_discrete(limits = rev) +
    scale_fill_continuous(high = "#e63946", low = "#F9C8CB", 
        labels = c("0%", "10%", "20%", "30%", "40%")) +
    # scale_fill_discrete(name = "Dose", labels = c("A", "B", "C"))
    tema 
    
ggsave(file = paste0(out_figs, "t_efectos", v_format), width = 10, height = 8)    

# Talba de las reacciones a las agresiones
ggplot(df_reaccion, aes(x = violencia, y = reaccion, fill = porcent)) +
    geom_tile(color = "gray90", size = 1.5) +
    geom_text(aes(label = text), family = v_fuente, color = "black", 
        size = rel(4)) +
    labs(title = "Tipos de reacciones ante las agresiones", 
        subtitle = "Por dimensión de violencia", 
        x = "", 
        y = "", 
        fill = "Frecuencia") +
    scale_x_discrete(position = "top") +
    scale_y_discrete(limits = rev) +
    scale_fill_continuous(high = "#e63946", low = "#F9C8CB", 
        labels = c("0%", "10%", "20%", "30%")) +
    tema

ggsave(file = paste0(out_figs, "t_reacciones", v_format), width = 10, height = 12)    


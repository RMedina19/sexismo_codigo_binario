#------------------------------------------------------------------------------#
# Proyecto:                   Tesis: Seximo en código binario 
# Objetivo:                   Generar tablas de calor de códigos
#
# Encargada:                  Regina Isabel Medina Rosales
# Correo:                     regina.medina@alumnos.cide.edu
# Fecha de creación:          21 de abril de 2021
# Última actualización:       21 de abril de 2021
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

df_crudo_agres  <- read_excel(paste0(inp_data, "/Co-ocurrencia-violencias-agresiones.xlsx"))
df_crudo_efect  <- read_excel(paste0(inp_data, "/Co-ocurrencia-violencias-efectos.xlsx"))
df_crudo_react  <- read_excel(paste0(inp_data, "/Co-ocurrencia-violencias-reacción.xlsx"))

# 02. Limpiar datos ------------------------------------------------------------

## 02.1. Tabla de las agresiones y el tipo de violencia ------------------------

# Nombres de las variables en la base cruda
v_names     <- names(df_crudo_agres)

# Limpiar nombres de las columnas y de los identificadores únicos de las agresiones
df_renamed <- df_crudo_agres                %>% 
    rename(id = v_names[1], 
            v_gen = v_names[2], 
            v_dig = v_names[3], 
            v_lab = v_names[4], 
            v_off = v_names[5], 
            v_pol = v_names[6])             %>% 
    mutate(id = str_sub(id, 2, 6))          %>% 
    select(id, v_gen, v_dig, v_pol)        


table(df_renamed$v_gen)
table(df_renamed$v_dig)
table(df_renamed$v_pol)

# Convertir frecuencia de los códigos en variables binarias
df_limpio <- df_renamed %>% 
    mutate(v_gen = if_else(v_gen == 0, 0, 1), 
            v_dig = if_else(v_dig == 0, 0, 1), 
            v_pol = if_else(v_pol == 0, 0, 1))
    
# Verificar que sea binario y que sumen 46 (el total de agresiones documentadas)
table(df_limpio$v_gen)
table(df_limpio$v_dig)
table(df_limpio$v_pol)

# Sacar combinaciones (excluyentes) de las posibles combinaciones de violencia
df_combinaciones <- df_limpio %>% 
    mutate(
        gen = if_else((v_gen == 1 & v_dig == 0 & v_pol == 0), 1, 0), 
        dig = if_else((v_gen == 0 & v_dig == 1 & v_pol == 0), 1, 0), 
        pol = if_else((v_gen == 0 & v_dig == 0 & v_pol == 1), 1, 0), 
        gen_dig = if_else((v_gen == 1 & v_dig == 1 & v_pol == 0), 1, 0), 
        gen_pol = if_else((v_gen == 1 & v_dig == 0 & v_pol == 1), 1, 0), 
        dig_pol = if_else((v_gen == 0 & v_dig == 1 & v_pol == 1), 1, 0), 
        gen_dig_pol = if_else((v_gen == 1 & v_dig == 1 & v_pol == 1), 1, 0)) %>% 
    mutate(tipo_violencia = case_when(
        gen == 1 ~ "Violencia de género", 
        dig == 1 ~ "Violencia digital", 
        pol == 1 ~ "Violencia política", 
        gen_dig == 1 ~ "Violencia de género y digital", 
        gen_pol == 1 ~ "Violencia de género y política", 
        dig_pol == 1 ~ "Violencia digital y política", 
        gen_dig_pol == 1 ~ "Violencia de género, digital y política")) %>% 
    group_by(tipo_violencia) %>% 
    summarise(total = n())


sum(df_combinaciones$gen)

## 02.2. Tabla de los efectos de las agresiones --------------------------------
# Nombres de las variables en la base cruda
v_names     <- names(df_crudo_efect)

# Limpiar nombres y variables
df_efecto_renamed   <- df_crudo_efect       %>% 
    rename(
            efecto  = v_names[1], 
            gen_abs = v_names[2], 
            dig_abs = v_names[3], 
            pol_abs = v_names[4])           %>% 
    mutate(# total = as.numeric(str_extract(efecto, "[[:digit:]]+")), # Poner frecuencia (números) en otra variable
        efecto = str_extract(efecto, "[[\\w\\s]]+"),    # Quitar números y caracteres especiales 
        efecto = str_sub(efecto, 2, -4))# Quitar "Gr" y espacios en blanco 


# Tabla con los porcentajes según el total de los efectos 
df_efecto <- df_efecto_renamed %>% 
    mutate(
        total_abs = rowSums(cbind(gen_abs, dig_abs, pol_abs)), 
        gen_por   = round(gen_abs  *100/total_abs, 1), 
        dig_por   = round(dig_abs *100/total_abs, 1),  
        pol_por   = round(pol_abs*100/total_abs, 1), 
        total_por = round(total_abs*100/total_abs, 0)) %>% 
    mutate(
        gen_text  = paste0(gen_por, "%\n (n=", gen_abs  , ")"), 
        dig_text  = paste0(dig_por, "%\n (n=", dig_abs , ")"), 
        pol_text  = paste0(pol_por, "%\n (n=", pol_abs, ")"), 
        total_text = paste0(total_por, "%\n (n=", total_abs,      ")"))

df_efecto_abs <- df_efecto              %>%
    select(efecto, gen_abs, dig_abs, pol_por, total_abs)

df_efecto_por <- df_efecto              %>%
    select(efecto, gen_por, dig_por, pol_por, total_por)

df_efecto_tex <- df_efecto              %>%
    select(efecto, gen_text, dig_text, pol_text, total_text)


# Tabla con los porcentajes según el total del tipo de violencia 

df_efecto2 <- df_efecto_renamed %>% 
    bind_rows(summarise_all(., funs(if(is.numeric(.)) sum(.) else "Total"))) %>% 
    mutate(gen_por   = round(gen_abs  *100/gen_abs[efecto   == "Total"], 1), 
            dig_por  = round(dig_abs *100/dig_abs[efecto  == "Total"], 1),  
            pol_por  = round(pol_abs*100/pol_abs[efecto == "Total"], 1))  %>% 
    mutate(gen_text  = paste0(gen_por, "%\n (n=", v_genero  , ")"), 
            dig_text = paste0(dig_por, "%\n (n=", v_digital , ")"), 
            pol_text = paste0(pol_por, "%\n (n=", v_politica, ")"))
   
df_efecto_abs2 <- df_efecto2             %>% 
    select(efecto, v_genero, v_digital, v_politica)

df_efecto_por2 <- df_efecto2             %>% 
    select(efecto, v_gen_por, v_dig_por, v_pol_por)

df_efecto_tex2 <- df_efecto2             %>% 
    select(efecto, v_gen_text, v_dig_text, v_pol_text)


## 02.3. Tabla de las reacciones a las agresiones ------------------------------

# Nombres de las variables en la 
v_names     <- names(df_crudo_react)

# Limpiar nombres 
df_reaccion_renamed <- df_crudo_react           %>% 
    rename(efecto  = v_names[1], 
        v_genero   = v_names[2], 
        v_digital  = v_names[3], 
        v_politica = v_names[4])        %>% 
    mutate(# total = as.numeric(str_extract(efecto, "[[:digit:]]+")), # Poner frecuencia (números) en otra variable
        efecto = str_extract(efecto, "[[\\w\\s]]+"),    # Quitar números y caracteres especiales 
        efecto = str_sub(efecto, 2, -4)) # Quitar "Gr" y espacios en blanco 


# Tabla con los porcentajes según el total de las reacciones 

df_reaccion <- df_reaccion_renamed      %>%         
        mutate(total  = rowSums(cbind(v_genero, v_digital, v_politica)), 
        v_gen_por = round(v_genero  *100/total, 1), 
        v_dig_por = round(v_digital *100/total, 1),  
        v_pol_por = round(v_politica*100/total, 1), 
        total_por = round(total*100/total, 0))  %>% 
    mutate(v_gen_text  = paste0(v_gen_por, "%\n (n=", v_genero  , ")"), 
        v_dig_text = paste0(v_dig_por, "%\n (n=", v_digital , ")"), 
        v_pol_text = paste0(v_pol_por, "%\n (n=", v_politica, ")"), 
        total_text = paste0(total_por, "%\n (n=", total,      ")"))

df_reaccion_abs <- df_reaccion              %>% 
    select(efecto, v_genero, v_digital, v_politica, total)

df_reaccion_por <- df_reaccion              %>% 
    select(efecto, v_gen_por, v_dig_por, v_pol_por, total_por)

df_reaccion_tex  <- df_reaccion              %>% 
    select(efecto, v_gen_text, v_dig_text, v_pol_text, total_text)



# Tabla con los porcentajes según el total del tipo de violencia 

df_reaccion2 <- df_reaccion_renamed %>% 
    bind_rows(summarise_all(., funs(if(is.numeric(.)) sum(.) else "Total"))) %>% 
    mutate(v_gen_por   = round(v_genero  *100/v_genero[efecto   == "Total"], 1), 
        v_dig_por  = round(v_digital *100/v_digital[efecto  == "Total"], 1),  
        v_pol_por  = round(v_politica*100/v_politica[efecto == "Total"], 1))  %>% 
    mutate(v_gen_text  = paste0(v_gen_por, "%\n (n=", v_genero  , ")"), 
        v_dig_text = paste0(v_dig_por, "%\n (n=", v_digital , ")"), 
        v_pol_text = paste0(v_pol_por, "%\n (n=", v_politica, ")"))

df_reaccion_abs2 <- df_efecto              %>% 
    select(efecto, v_genero, v_digital, v_politica)

df_reaccion_por2 <- df_efecto              %>% 
    select(efecto, v_gen_por, v_dig_por, v_pol_por)

df_reaccion_tex2 <- df_efecto              %>% 
    select(efecto, v_gen_text, v_dig_text, v_pol_text)

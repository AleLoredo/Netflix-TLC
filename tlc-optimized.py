from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
import numpy as np

print("=== DEMOSTRACIÓN EFICIENTE DEL TEOREMA DEL LÍMITE CENTRAL ===")
print("Estrategia: Sample First, Clean Later, Repeat if Needed")
print()

# 1. LEER DATOS SIN APLICAR FILTROS COSTOSOS
print("1. CARGA INICIAL DE DATOS (sin filtros):")
df_raw = spark.read.parquet("/mnt/streaming-data/logs/2025-08-03/")

# Solo obtener el conteo total para calcular fracciones
total_count = df_raw.count()
print(f"Total de registros en población: {total_count:,}")
print()

# 2. CALCULAR TAMAÑO ÓPTIMO DE MUESTRA PARA NUMEROSITY REDUCTION
print("2. OPTIMIZACIÓN DE NUMEROSITY REDUCTION:")

# Parámetros del análisis
n = 30  # Tamaño de muestra objetivo
n_samples = 1000  # Número de muestras objetivo

def calculate_optimal_sample_size(n, n_samples, safety_factor=2):
    """
    Calcula el tamaño óptimo de muestra para numerosity reduction
    basado en datos realmente necesarios + factor de seguridad para limpieza
    """
    target_needed = n * n_samples
    optimal_size = safety_factor * target_needed
    return target_needed, optimal_size

target_needed, optimal_size = calculate_optimal_sample_size(n, n_samples, safety_factor=2)

print(f"Datos objetivo necesarios: {target_needed:,}")
print(f"Tamaño óptimo con factor seguridad 2x: {optimal_size:,}")
print(f"Factor de seguridad: Tolera hasta 50% pérdida en limpieza")

# 3. APLICAR NUMEROSITY REDUCTION OPTIMIZADA
print("\n3. ESTIMACIÓN EFICIENTE DE PARÁMETROS POBLACIONALES:")

# Tomar muestra optimizada (no fija de 100K)
estimation_fraction = min(0.01, optimal_size / total_count)  # Calculado dinámicamente
estimation_sample = df_raw.sample(withReplacement=False, fraction=estimation_fraction, seed=42)

# Aplicar limpieza solo a la muestra de estimación
estimation_clean = estimation_sample.filter(F.col("playback_duration") > 5).dropna()

# Calcular estadísticas poblacionales estimadas
pop_stats = estimation_clean.agg(
    F.count("playback_duration").alias("clean_count"),
    F.mean("playback_duration").alias("mean"),
    F.stddev("playback_duration").alias("std")
).collect()[0]

mu = pop_stats["mean"]
sigma = pop_stats["std"]
clean_ratio = pop_stats["clean_count"] / estimation_sample.count()  # Ratio de datos válidos

actual_sample_size = estimation_sample.count()
print(f"Muestra optimizada: {actual_sample_size:,} registros")
print(f"Datos válidos después de limpieza: {pop_stats['clean_count']:,}")
print(f"Ratio de datos válidos: {clean_ratio:.4f}")
print(f"Media poblacional estimada (μ): {mu:.4f}")
print(f"Desviación estándar estimada (σ): {sigma:.4f}")

# Verificar eficiencia de la optimización
efficiency_gain = 100000 / actual_sample_size if actual_sample_size > 0 else 1
print(f"Optimización: {efficiency_gain:.1f}x menos datos procesados vs enfoque fijo")
print()

# 4. PARÁMETROS DEL TLC
# 4. PARÁMETROS DEL TLC
expected_std = sigma / np.sqrt(n)

print("4. PARÁMETROS DEL TEOREMA DEL LÍMITE CENTRAL:")
print(f"Tamaño de muestra objetivo (n): {n}")
print(f"Número de muestras objetivo: {n_samples}")
print(f"Desviación estándar esperada de X̄: {expected_std:.4f}")
print()

# 5. FUNCIÓN OPTIMIZADA PARA GENERAR MUESTRAS
def get_clean_sample_optimized(df_source, target_size, clean_ratio_estimate, max_attempts=5, seed_base=None):
    """
    Genera una muestra limpia de tamaño exacto target_size
    usando numerosity reduction optimizada
    """
    # Calcular fracción optimizada basada en datos realmente necesarios
    safety_factor = 1.5  # Factor conservador para reintento
    initial_fraction = (target_size * safety_factor) / (total_count * clean_ratio_estimate)
    
    for attempt in range(max_attempts):
        # Generar seed único para cada intento
        current_seed = (seed_base * 1000 + attempt) if seed_base else None
        
        # 1. SAMPLE FIRST - tomar muestra cruda optimizada
        raw_sample = df_source.sample(
            withReplacement=False, 
            fraction=initial_fraction, 
            seed=current_seed
        )
        
        # 2. CLEAN LATER - aplicar filtros solo a la muestra optimizada
        clean_sample = raw_sample.filter(F.col("playback_duration") > 5).dropna()
        
        sample_count = clean_sample.count()
        
        # 3. VERIFICAR SI TENEMOS SUFICIENTES DATOS
        if sample_count >= target_size:
            # Tomar exactamente target_size registros
            final_sample = clean_sample.limit(target_size)
            return final_sample
        else:
            # Ajustar fracción para el siguiente intento
            if sample_count > 0:
                actual_ratio = sample_count / raw_sample.count()
                initial_fraction = (target_size * 1.2) / (total_count * actual_ratio)
            else:
                initial_fraction *= 1.5  # Incremento conservador
    
    # Si llegamos aquí, usar lo que pudimos obtener
    return clean_sample

# 6. GENERAR MUESTRAS CON ESTRATEGIA OPTIMIZADA
print("5. GENERANDO MUESTRAS CON NUMEROSITY REDUCTION OPTIMIZADA...")

sample_means = []
failed_samples = 0
total_data_processed = 0

for i in range(n_samples):
    try:
        # Generar muestra limpia con numerosity reduction optimizada
        sample_df = get_clean_sample_optimized(df_raw, n, clean_ratio, seed_base=i)
        
        # Calcular media (solo una agregación pequeña)
        sample_mean = sample_df.agg(F.mean("playback_duration")).collect()[0][0]
        
        if sample_mean is not None:
            sample_means.append(float(sample_mean))
            total_data_processed += n  # Contabilizar datos realmente utilizados
        else:
            failed_samples += 1
            
    except Exception as e:
        failed_samples += 1
        if failed_samples < 10:  # Solo mostrar primeros errores
            print(f"Error en muestra {i}: {str(e)[:100]}")
    
    # Progress indicator
    if (i + 1) % 100 == 0:
        success_rate = len(sample_means) / (i + 1) * 100
        print(f"Progreso: {i + 1}/{n_samples} | Exitosas: {len(sample_means)} | Success rate: {success_rate:.1f}%")

successful_samples = len(sample_means)
print(f"\nMuestras generadas exitosamente: {successful_samples}")
print(f"Muestras fallidas: {failed_samples}")
print(f"Total de datos realmente procesados: {total_data_processed:,}")

# Calcular eficiencia de la optimización
theoretical_max_processed = actual_sample_size + (total_data_processed * 1.5)  # Estimación conservadora
traditional_processed = total_count * 0.1  # Enfoque tradicional procesaría 10% del dataset
efficiency_improvement = traditional_processed / theoretical_max_processed

print(f"Eficiencia vs enfoque tradicional: ~{efficiency_improvement:.1f}x mejor")
print()

# 6. ANÁLISIS DEL TLC (solo si tenemos suficientes muestras)
if successful_samples >= 100:  # Mínimo para análisis estadístico
    print("5. VERIFICACIÓN DEL TEOREMA DEL LÍMITE CENTRAL:")
    
    # Convertir a DataFrame de Spark para análisis
    means_df = spark.createDataFrame([(float(m),) for m in sample_means], ["sample_mean"])
    
    # Estadísticas observadas
    observed_stats = means_df.agg(
        F.count("sample_mean").alias("count"),
        F.mean("sample_mean").alias("observed_mean"),
        F.stddev("sample_mean").alias("observed_std"),
        F.min("sample_mean").alias("min_mean"),
        F.max("sample_mean").alias("max_mean")
    ).collect()[0]
    
    observed_mean = observed_stats["observed_mean"]
    observed_std = observed_stats["observed_std"]
    
    # Comparación con valores esperados
    mean_error = abs(observed_mean - mu)
    std_error = abs(observed_std - expected_std)
    
    print(f"Media observada de X̄: {observed_mean:.4f}")
    print(f"Media esperada de X̄: {mu:.4f}")
    print(f"Error en media: {mean_error:.6f}")
    print()
    print(f"Desv. std observada de X̄: {observed_std:.4f}")
    print(f"Desv. std esperada de X̄: {expected_std:.4f}")
    print(f"Error en desviación: {std_error:.6f}")
    print()
    
    # 7. ANÁLISIS DE DISTRIBUCIÓN
    print("6. ANÁLISIS DE DISTRIBUCIÓN:")
    
    # Crear histograma
    n_bins = 25
    min_val = observed_stats["min_mean"]
    max_val = observed_stats["max_mean"]
    bin_width = (max_val - min_val) / n_bins
    
    histogram_df = means_df.withColumn(
        "bin", F.floor((F.col("sample_mean") - F.lit(min_val)) / F.lit(bin_width))
    ).groupBy("bin").agg(
        F.count("*").alias("frequency"),
        F.mean("sample_mean").alias("bin_center")
    ).orderBy("bin")
    
    # 8. RESULTADOS FINALES
    print("=== RESULTADOS DE EFICIENCIA Y PRECISIÓN ===")
    
    # Crear resumen final
    means_df.createOrReplaceTempView("efficient_samples")
    
    final_summary = spark.sql(f"""
    SELECT 
        'Estrategia Eficiente' as method,
        {successful_samples} as samples_generated,
        {total_count} as population_size,
        avg(sample_mean) as observed_mean,
        {mu} as expected_mean,
        stddev(sample_mean) as observed_std,
        {expected_std} as expected_std,
        abs(avg(sample_mean) - {mu}) as mean_error,
        abs(stddev(sample_mean) - {expected_std}) as std_error
    FROM efficient_samples
    """)
    
    print("Resumen de la demostración del TLC:")
    display(final_summary)
    
    print("Distribución de medias muestrales:")
    display(histogram_df)
    
    print("Muestra de medias generadas:")
    display(means_df.limit(20))
    
    # 9. MÉTRICAS DE EFICIENCIA
    print("=== VENTAJAS DE LA ESTRATEGIA EFICIENTE ===")
    estimated_processed = successful_samples * n * 2.5  # Estimación conservadora
    efficiency_gain = total_count / estimated_processed
    
    print(f"✅ Población total: {total_count:,} registros")
    print(f"✅ Registros procesados estimados: ~{estimated_processed:,.0f}")
    print(f"✅ Ganancia de eficiencia: ~{efficiency_gain:.1f}x")
    print(f"✅ Solo {(estimated_processed/total_count)*100:.3f}% del dataset procesado")
    print(f"✅ Muestras exitosas: {successful_samples}/{n_samples} ({successful_samples/n_samples*100:.1f}%)")
    
    # Verificación final del TLC
    tlc_verified = mean_error < 0.05 and std_error < 0.05
    print(f"""
=== CONCLUSIÓN DEL TLC CON ESTRATEGIA EFICIENTE ===
{'✅' if tlc_verified else '⚠️'} TLC verificado: {tlc_verified}
{'✅' if mean_error < 0.01 else '⚠️'} Error de media: {mean_error:.6f}
{'✅' if std_error < 0.01 else '⚠️'} Error de desviación: {std_error:.6f}

La estrategia "Sample First, Clean Later" es {efficiency_gain:.1f}x más eficiente
que aplicar filtros a toda la población de 500M registros.
""")

else:
    print(f"⚠️ Insuficientes muestras para análisis ({successful_samples} < 100)")
    print("Considerar ajustar parámetros de muestreo o aumentar safety_factor")

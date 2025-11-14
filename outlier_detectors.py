import numpy as np
import pandas as pd
try:
    import changefinder
except ImportError:
    changefinder = None
    print("Warning: changefinder not installed. Install with: pip install changefinder")


class OutlierDetector:
    """Clase base para detectores de outliers"""
    def detect(self, data, residuals=None):
        """
        Detecta outliers
        
        Args:
            data: Serie de tiempo original
            residuals: Residuos del modelo (opcional, para algunos detectores)
            
        Returns:
            labels: Array de etiquetas
            scores: Scores relevantes
            additional_data: Datos adicionales (ej: valores sin outliers)
        """
        raise NotImplementedError


class AdaptiveVarianceDetector(OutlierDetector):
    """
    Detector basado en varianza adaptativa y ChangeFinder.
    Asume que los residuos distribuyen normal.
    """
    def __init__(
        self,
        alpha=0.005,
        quantile=0.995,
        factor_olvido=0.02,
        lag_cambio=2,
        suavizado=7,
        change_quantile=0.99
    ):
        """
        Args:
            alpha: Factor de suavizado para varianza adaptativa
            quantile: Cuantil para threshold de outliers
            factor_olvido: Factor de olvido para ChangeFinder
            lag_cambio: Orden AR para ChangeFinder
            suavizado: Ventana de suavizado para ChangeFinder
            change_quantile: Cuantil para threshold de cambios
        """
        self.alpha = alpha
        self.quantile = quantile
        self.factor_olvido = factor_olvido
        self.lag_cambio = lag_cambio
        self.suavizado = suavizado
        self.change_quantile = change_quantile
        
        if changefinder is None:
            raise ImportError("changefinder is required. Install with: pip install changefinder")
    
    def detect(self, data, residuals=None):
        """Detecta outliers usando varianza adaptativa y ChangeFinder"""
        if residuals is None:
            raise ValueError("AdaptiveVarianceDetector requiere residuals")
        
        n = len(data)
        
        # 1. Calcular varianza adaptativa
        serie_residuos = pd.Series(residuals)
        var_inicial = np.nanvar(serie_residuos.values)
        if not np.isfinite(var_inicial) or var_inicial <= 0:
            var_inicial = 1.0
        
        s2 = []
        s2_curr = var_inicial
        for r in serie_residuos.values:
            if np.isfinite(r):
                s2_curr = (1 - self.alpha) * s2_curr + self.alpha * (r**2)
            s2.append(max(s2_curr, 1e-8))
        s2 = np.array(s2, dtype=float)
        
        # 2. Calcular outlier score (log-likelihood negativa)
        outlier_score = 0.5 * (np.log(2 * np.pi * s2) + (residuals**2) / s2)
        
        # 3. Threshold para outliers
        valid_out = np.isfinite(outlier_score)
        thr_out = np.quantile(outlier_score[valid_out], self.quantile) if valid_out.any() else np.inf
        is_out = outlier_score >= thr_out
        
        # 4. ChangeFinder para detectar cambios
        cf = changefinder.ChangeFinder(
            r=self.factor_olvido, 
            order=self.lag_cambio, 
            smooth=self.suavizado
        )
        score_cambio = np.array([cf.update(float(v)) for v in data], dtype=float)
        
        # 5. Threshold para cambios
        valid_ch = np.isfinite(score_cambio)
        thr_ch = np.quantile(score_cambio[valid_ch], self.change_quantile) if valid_ch.any() else np.inf
        is_ch = score_cambio >= thr_ch
        
        # 6. Generar labels
        labels = np.full(n, "normal", dtype=object)
        labels[is_out] = "outlier"
        labels[is_ch] = "change"
        
        return {
            'labels': labels,
            'outlier_score': outlier_score,
            'change_score': score_cambio,
            'residuals': residuals
        }

class DiffDetector(OutlierDetector):
    """
    Detector basado en diferencias centradas.
    """
    def __init__(self, lambda_centrada=None, k=0, quantile_low=0.01, quantile_high=0.999):
        """
        Args:
            lambda_centrada: Multiplicador de std para threshold de diff_centrada
            k: Threshold para diferencia absoluta con dato anterior (si 0, se calcula desde cuantiles)
            quantile_low: Cuantil inferior para calcular k
            quantile_high: Cuantil superior para calcular k
        """
        self.lambda_centrada = lambda_centrada
        self.k = k
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
    
    def conseguir_diff_centrada(self, data):
        """
        Calcula la diferencia centrada para cada punto.
        Versión vectorizada.
        """
        serie = pd.Series(data)
        
        val_prev = serie.shift(1).fillna(serie.shift(2))
        val_sig = serie.shift(-1).fillna(serie.shift(-2))
        
        diff_prev = np.abs(val_prev - serie)
        diff_sig = np.abs(val_sig - serie)
        
        diff_centrada = np.minimum(diff_prev, diff_sig)
        
        return diff_centrada.values
    
    def detect(self, data, residuals=None):
        """Detecta outliers usando diferencias centradas - VECTORIZADO"""
        n = len(data)
        serie = pd.Series(data)
        
        # Calcular diff_centrada de toda la serie
        diff_centrada_total = self.conseguir_diff_centrada(data)
        
        # Calcular STD de diff_centrada
        valid_diff = diff_centrada_total[np.isfinite(diff_centrada_total)]
        if len(valid_diff) > 0:
            std_diff_centrada = np.std(valid_diff)
        else:
            std_diff_centrada = 1.0
        
        # Lambda_centrada por defecto si no se especifica
        if self.lambda_centrada is None or self.lambda_centrada == 0:
            lambda_centrada = 12  # Valor por defecto
        else:
            lambda_centrada = self.lambda_centrada
        
        # Calcular k si es 0
        if self.k == 0:
            valid_serie = serie[np.isfinite(serie)]
            if len(valid_serie) > 0:
                k = np.quantile(valid_serie, self.quantile_high) - np.quantile(valid_serie, self.quantile_low)
            else:
                k = 1.0
        else:
            k = self.k
        
        # Calcular threshold
        threshold_diff_centrada = lambda_centrada * std_diff_centrada
        
        print(f"    STD diff_centrada: {std_diff_centrada:.4f}")
        print(f"    Lambda centrada: {lambda_centrada:.4f}")
        print(f"    Threshold (lambda * std): {threshold_diff_centrada:.4f}")
        print(f"    K: {k:.4f}")
        
        # Inicializar
        valores_sin_outliers = data.copy()
        labels = np.full(n, "normal", dtype=object)
        
        # Iteración con reemplazo de outliers
        max_iterations = 5  # Máximo de iteraciones para converger
        
        for iteration in range(max_iterations):
            # Calcular diff_centrada y diff sobre valores actuales (sin outliers)
            diff_centrada = self.conseguir_diff_centrada(valores_sin_outliers)
            diff = pd.Series(valores_sin_outliers).diff().values
            
            # Recalcular std en cada iteración
            valid_diff_iter = diff_centrada[np.isfinite(diff_centrada)]
            if len(valid_diff_iter) > 0:
                std_iter = np.std(valid_diff_iter)
            else:
                std_iter = std_diff_centrada
            
            threshold_iter = lambda_centrada * std_iter
            
            # Detectar outliers
            is_outlier_diff_centrada = diff_centrada >= threshold_iter
            is_outlier_diff = np.abs(diff) >= k
            
            is_outlier = is_outlier_diff_centrada | is_outlier_diff
            
            # Marcar NaN
            is_outlier[np.isnan(data)] = False
            
            # Si no hay cambios, terminar
            if not np.any(is_outlier):
                break
            
            # Actualizar labels
            labels[is_outlier] = "outlier"
            
            # Reemplazar outliers con valor anterior + ruido
            for i in np.where(is_outlier)[0]:
                if i > 0:
                    val_prev = valores_sin_outliers[i-1]
                    ruido = np.random.normal(0, 0.01 * abs(val_prev) if val_prev != 0 else 0.01)
                    valores_sin_outliers[i] = val_prev + ruido
                else:
                    # Primer valor: buscar siguiente válido
                    siguiente_valido = None
                    for j in range(i+1, min(i+10, n)):
                        if not np.isnan(data[j]):
                            siguiente_valido = data[j]
                            break
                    if siguiente_valido is not None:
                        valores_sin_outliers[i] = siguiente_valido
        
        # Calcular scores finales
        diff_centrada_final = self.conseguir_diff_centrada(valores_sin_outliers)
        diff_final = pd.Series(valores_sin_outliers).diff().values
        
        # Marcar NaN en labels
        labels[np.isnan(data)] = np.nan
        
        return {
            'labels': labels,
            'diff_centrada_score': diff_centrada_final,
            'diff_score': np.abs(diff_final),
            'valores_sin_outliers': valores_sin_outliers,
            'lambda_centrada_usado': lambda_centrada,
            'std_usado': std_diff_centrada,
            'threshold_usado': threshold_diff_centrada,
            'k_usado': k
        }
/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.query.request;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.core.ParseField;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.*;
import static org.opensearch.knn.index.query.KNNQueryBuilder.*;

/**
 * MethodParameters are engine and algorithm related parameters that clients can pass in knn query
 * This enum holds metadata which helps parse and have basic validation related to MethodParameter
 */
@Getter
@RequiredArgsConstructor
public enum MethodParameter {

    EF_SEARCH(METHOD_PARAMETER_EF_SEARCH, Version.V_2_16_0, EF_SEARCH_FIELD) {
        @Override
        public Integer parse(Object value) {
            return parseInteger(value, METHOD_PARAMETER_EF_SEARCH);
        }

        @Override
        public ValidationException validate(Object value) {
            final Integer ef = parse(value);
            if (ef != null && ef > 0) {
                return null;
            }

            ValidationException validationException = new ValidationException();
            validationException.addValidationError(METHOD_PARAMETER_EF_SEARCH + " should be greater than 0");
            return validationException;
        }
    },

    NPROBE(METHOD_PARAMETER_NPROBES, Version.V_2_16_0, NPROBE_FIELD) {
        @Override
        public Integer parse(Object value) {
            return parseInteger(value, METHOD_PARAMETER_EF_SEARCH);
        }

        @Override
        public ValidationException validate(Object value) {
            final Integer nprobe = parse(value);
            if (nprobe != null && nprobe > 0) {
                return null;
            }

            ValidationException validationException = new ValidationException();
            validationException.addValidationError(METHOD_PARAMETER_NPROBES + " should be greater than 0");
            return validationException;
        }
    },

    OVERQUERY_FACTOR(METHOD_PARAMETER_OVERQUERY_FACTOR, Version.V_3_0_0, OVERQUERY_FACTOR_FIELD) {
        @Override
        public Integer parse(Object value) {
            return parseInteger(value, METHOD_PARAMETER_OVERQUERY_FACTOR);
        }

        @Override
        public ValidationException validate(Object value) {
            final Integer parsed = parse(value);
            if (parsed != null && parsed > 0) {
                return null;
            }

            ValidationException validationException = new ValidationException();
            validationException.addValidationError(METHOD_PARAMETER_OVERQUERY_FACTOR + " should be greater than 0");
            return validationException;
        }
    },

    THRESHOLD(METHOD_PARAMETER_THRESHOLD, Version.V_3_0_0, THRESHOLD_FIELD) {
        @Override
        public Double parse(Object value) {
            return parseDouble(value, METHOD_PARAMETER_THRESHOLD);
        }

        @Override
        public ValidationException validate(Object value) {
            final Double parsed = parse(value);
            if (parsed != null && parsed >= 0) {
                return null;
            }

            ValidationException validationException = new ValidationException();
            validationException.addValidationError(METHOD_PARAMETER_THRESHOLD + " should be greater than or equal to 0");
            return validationException;
        }
    },

    RERANK_FLOOR(METHOD_PARAMETER_RERANK_FLOOR, Version.V_3_0_0, REREANK_FLOOR_FIELD) {
        @Override
        public Double parse(Object value) {
            return parseDouble(value, METHOD_PARAMETER_RERANK_FLOOR);
        }

        @Override
        public ValidationException validate(Object value) {
            final Double parsed = parse(value);
            if (parsed != null && parsed >= 0) {
                return null;
            }

            ValidationException validationException = new ValidationException();
            validationException.addValidationError(METHOD_PARAMETER_RERANK_FLOOR + " should be greater than or equal to 0");
            return validationException;
        }
    },
    USE_PRUNING(METHOD_PARAMETER_USE_PRUNING, Version.V_3_0_0, USE_PRUNING_FIELD) {
        @Override
        public Boolean parse(Object value) {
            return parseBoolean(value, METHOD_PARAMETER_USE_PRUNING);
        }

        @Override
        public ValidationException validate(Object value) {
            return null;
        }
    },;

    private final String name;
    private final Version version;
    private final ParseField parseField;

    private static Map<String, MethodParameter> PARAMETERS_DIR;

    public abstract <T> T parse(Object value);

    // These are preliminary validations on rest layer
    public abstract ValidationException validate(Object value);

    public static MethodParameter enumOf(final String name) {
        if (PARAMETERS_DIR == null) {
            PARAMETERS_DIR = new HashMap<>();
            for (final MethodParameter methodParameter : MethodParameter.values()) {
                PARAMETERS_DIR.put(methodParameter.name, methodParameter);
            }
        }
        return PARAMETERS_DIR.get(name);
    }

    private static Integer parseInteger(Object value, String name) {
        try {
            return Integer.parseInt(String.valueOf(value));
        } catch (final NumberFormatException e) {
            throw new IllegalArgumentException(name + " value must be an integer");
        }
    }

    private static Double parseDouble(Object value, String name) {
        try {
            return Double.parseDouble(String.valueOf(value));
        } catch (final NumberFormatException e) {
            throw new IllegalArgumentException(name + " value must be a double");
        }
    }

    private static Boolean parseBoolean(Object value, String name) {
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        throw new IllegalArgumentException(name + " value must be a boolean");
    }
}

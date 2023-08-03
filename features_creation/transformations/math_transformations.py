from .base_transformations import Transformations


class MathTransformations(Transformations):
    _symmetric_ops = ["hypotenuse", "mean", "prod", "sum"]
    _non_symmetric_ops = ["floordiv", "mod", "rest", "truediv"]

    def generate_transformations(self, _, columns):
        transformations = {}

        for i_1 in range(len(columns)):
            for i_2 in range(i_1 + 1, len(columns)):
                # non symmetric operations
                for op in self._non_symmetric_ops:
                    col_name = f"{columns[i_1]}__{op}__{columns[i_2]}"
                    transformations[col_name] = {
                        "col_1": columns[i_1],
                        "col_2": columns[i_2],
                        "op": op,
                        "type": "math",
                    }

                if i_1 == i_2:
                    continue

                # symmetric operations
                for op in self._symmetric_ops:
                    col_name = f"{columns[i_1]}__{op}__{columns[i_2]}"
                    transformations[col_name] = {
                        "col_1": columns[i_1],
                        "col_2": columns[i_2],
                        "op": op,
                        "type": "math",
                    }

        return transformations

    def apply_transformations(self, df, **kwargs):
        col_1_name, col_2_name, op = kwargs["col_1"], kwargs["col_2"], kwargs["op"]
        col_1 = df[col_1_name]
        col_2 = df[col_2_name]

        # symmetric operations
        if op == "hypotenuse":
            return (col_1**2 + col_2**2) ** 0.5

        elif op == "mean":
            return (col_1 + col_2) / 2

        elif op == "prod":
            return col_1 * col_2

        if op == "sum":
            return col_1 + col_2

        # non symmetric operations

        elif op == "floordiv":
            return col_1 // col_2

        elif op == "mod":
            return col_1 % col_2

        elif op == "rest":
            return col_1 - col_2

        elif op == "truediv":
            return col_1 / col_2

        raise Exception("Operation not supported")

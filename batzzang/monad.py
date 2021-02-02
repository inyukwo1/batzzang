from typing import Callable, Type, Dict, Union, NewType, List, Any
from abc import ABC
from .lazy_modules import LazyModule
from . import timer

class State(ABC):
    pass


StateChecker = NewType("StateChecker", Callable[[State], bool])
TensorChain = NewType(
    "TensorChain",
    Callable[[State, Any], Any],
)


class Do:
    def Then(self, tensorchain: TensorChain) -> "Do":
        return self.__call__(tensorchain)

    def __init__(self, tensorchain: TensorChain):
        self.tensor_chains = [tensorchain]

    def __call__(self, tensorchain: TensorChain) -> "Do":
        self.tensor_chains.append(tensorchain)
        return self

    @timer.measure_time
    def bind(self, state_iter, list_prev_return):
        for tensor_chain in self.tensor_chains:
            list_prev_return = self._invoke_tensorchain(
                tensor_chain, list(state_iter), list_prev_return
            )

        return list_prev_return

    def _invoke_tensorchain(
        self, callback: TensorChain, states: List[State], list_prev_return: List,
    ):
        with timer.Pause() :
            list_new_return = [
                callback(state, tensor_dict)
                for state, tensor_dict in zip(states, list_prev_return)
            ]
        LazyModule.wait_all()
        return list_new_return


class If:
    def Then(self, tensorchain: TensorChain) -> "If":
        return self.__call__(tensorchain)

    def __init__(self, state_checker: Type[StateChecker]):
        self.state_checker = state_checker
        self.tensor_chains = []

    def __call__(self, tensorchain: TensorChain) -> "If":
        self.tensor_chains.append(tensorchain)
        return self

    @timer.measure_time
    def bind(self, state_iter, list_prev_return):
        true_indices = [
            idx for idx, state in enumerate(state_iter) if self.state_checker(state)
        ]
        checked_states = [
            state for idx, state in enumerate(state_iter) if idx in true_indices
        ]
        checked_prev_return = [
            prev_return
            for idx, prev_return in enumerate(list_prev_return)
            if idx in true_indices
        ]

        for tensor_chain in self.tensor_chains:
            tmp = self._invoke_tensorchain(
                tensor_chain, checked_states, checked_prev_return
            )
            checked_prev_return = tmp

        for idx, new_return_value in zip(true_indices, checked_prev_return):
            list_prev_return[idx] = new_return_value

        return list_prev_return

    def _invoke_tensorchain(
        self, callback: TensorChain, states: List[State], list_prev_return: List,
    ):
        with timer.Pause():
            list_new_return = [
                callback(state, tensor_dict)
                for state, tensor_dict in zip(states, list_prev_return)
            ]
        LazyModule.wait_all()
        return list_new_return


class While:
    @classmethod
    def Any(cls, stop_if_all_false: StateChecker, logic_unit: Union[Do, If]) -> "While":
        return While(stop_if_all_false, logic_unit)

    def __init__(self, stop_if_all_false: StateChecker, logic_unit: Union[Do, If]):
        self.stop_if_all_false = stop_if_all_false
        self.logic_units = [logic_unit]

    def __call__(self, logic_unit: If):
        self.logic_units.append(logic_unit)
        return self

    @timer.measure_time
    def bind(self, state_iter, prev_return):
        state_list = [state for state in state_iter]

        def is_not_done():
            check_results = [self.stop_if_all_false(state) for state in state_list]
            return any(check_results)

        while is_not_done():
            for idx, logic_unit in enumerate(self.logic_units):
                prev_return = logic_unit.bind(state_list, prev_return)
        return prev_return


class ForEachState:
    def __init__(self, state_iter):
        self.states = [state for state in state_iter]
        self.prev_return = [None] * len(self.states)

    @timer.measure_time
    def apply(self, logic: Union[While, If]) -> "ForEachState":
        if isinstance(logic, While):
            self.prev_return = logic.bind(self.states, self.prev_return)
        else:
            self.prev_return = logic.bind(self.states, self.prev_return)
        return self


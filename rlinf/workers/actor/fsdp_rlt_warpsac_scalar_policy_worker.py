# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.actor.fsdp_rlt_td3_policy_worker import RLTTD3FSDPPolicy


class RLTWarpSACScalarFSDPPolicy(RLTTD3FSDPPolicy):
    """RLT WarpSAC critic update with the current TD3 actor.

    Stage 3 keeps the TD3 actor objective, replay data flow, and twin scalar
    Q heads unchanged. The only algorithmic change is that the bootstrap
    action is sampled from the online actor while the bootstrap value is
    evaluated by the target critic.
    """

    def _next_actions_for_critic_target(self, next_obs):
        return self.model(
            forward_type=ForwardType.SAC,
            obs=next_obs,
        )

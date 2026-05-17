import * as THREE from "three";
import { Pose } from "kalidokit";

const BONE_ALIASES = {
  Hips:         ["hips", "pelvis"],
  Spine:        ["spine"],
  Spine1:       ["spine1"],
  Spine2:       ["spine2", "chest"],
  Neck:         ["neck"],
  Head:         ["head"],
  LeftArm:      ["leftarm", "leftupperarm"],
  LeftForeArm:  ["leftforearm", "leftlowerarm"],
  RightArm:     ["rightarm", "rightupperarm"],
  RightForeArm: ["rightforearm", "rightlowerarm"],
  LeftUpLeg:    ["leftupleg", "leftupperleg", "leftthigh"],
  LeftLeg:      ["leftleg", "leftlowerleg", "leftcalf"],
  LeftFoot:     ["leftfoot"],
  RightUpLeg:   ["rightupleg", "rightupperleg", "rightthigh"],
  RightLeg:     ["rightleg", "rightlowerleg", "rightcalf"],
  RightFoot:    ["rightfoot"],
  LeftHand:     ["lefthand"],
  RightHand:    ["righthand"],
};

function normalizeName(name) {
  return name.toLowerCase().replace(/mixamorig\d*/i, "").replace(/[._\-\s]/g, "");
}

class KalidokitController {
  constructor() {
    this.bones = {};
    this.avatar = null;
    this.lerpAmount = 0.3;
  }

  bindAvatar(avatar) {
    this.avatar = avatar.scene || avatar;
    this.bones = {};

    const allBones = [];
    this.avatar.traverse((obj) => {
      if (obj.isBone || obj.type === "Bone") allBones.push(obj);
    });

    for (const [slot, aliases] of Object.entries(BONE_ALIASES)) {
      for (const bone of allBones) {
        const n = normalizeName(bone.name);
        if (aliases.some((a) => n === a)) { this.bones[slot] = bone; break; }
      }
      if (!this.bones[slot]) {
        for (const bone of allBones) {
          const n = normalizeName(bone.name);
          if (aliases.some((a) => n.includes(a))) { this.bones[slot] = bone; break; }
        }
      }
    }

    console.log(`[Kalidokit] 바인딩: ${Object.keys(this.bones).length}/${Object.keys(BONE_ALIASES).length}`);
    Object.entries(this.bones).forEach(([s, b]) => console.log(`  ${s} → ${b.name}`));
  }

  _rigRotation(slot, rotation, dampener = 1, lerpAmount = 0.3) {
    const bone = this.bones[slot];
    if (!bone || !rotation) return;

    const euler = new THREE.Euler(
      rotation.x * dampener,
      rotation.y * dampener,
      rotation.z * dampener
    );
    const targetQuat = new THREE.Quaternion().setFromEuler(euler);
    bone.quaternion.slerp(targetQuat, lerpAmount);
  }

  update(worldLandmarks, normalizedLandmarks, video) {
    if (!worldLandmarks || !normalizedLandmarks || !this.avatar) return;

    const rig = Pose.solve(worldLandmarks, normalizedLandmarks, {
      runtime: "mediapipe",
      video,
    });
    if (!rig) return;

    const t = this.lerpAmount;

    if (rig.Hips) {
      this._rigRotation("Hips", rig.Hips.rotation, 0.7, t);
    }

    this._rigRotation("Spine",  rig.Spine, 0.45, t * 0.7);
    this._rigRotation("Spine1", rig.Spine, 0.35, t * 0.7);
    this._rigRotation("Spine2", rig.Spine, 0.25, t * 0.7);

    this._rigRotation("Neck", rig.Neck, 0.7, t * 0.6);
    this._rigRotation("Head", rig.Head, 0.7, t * 0.6);

    this._rigRotation("LeftArm",      rig.LeftUpperArm,  1, t);
    this._rigRotation("LeftForeArm",  rig.LeftLowerArm,  1, t);
    this._rigRotation("RightArm",     rig.RightUpperArm, 1, t);
    this._rigRotation("RightForeArm", rig.RightLowerArm, 1, t);
    this._rigRotation("LeftHand",     rig.LeftHand,  1, t);
    this._rigRotation("RightHand",    rig.RightHand, 1, t);

    this._rigRotation("LeftUpLeg",  rig.LeftUpperLeg,  1, t);
    this._rigRotation("LeftLeg",    rig.LeftLowerLeg,  1, t);
    this._rigRotation("RightUpLeg", rig.RightUpperLeg, 1, t);
    this._rigRotation("RightLeg",   rig.RightLowerLeg, 1, t);
  }

  setLerpAmount(v) { this.lerpAmount = v; }
}

export default KalidokitController;

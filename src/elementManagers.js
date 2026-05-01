import * as THREE from "three";
import { Landmark, VirtualLandmark, Bone, VirtualBone } from './elements.js'
export { PoseElementManager, BoneManager, VirtualBoneManager };

class PoseElementManager {
    constructor() {
        this.container = [];
        this.initKalmanFilter = () => { for (let element of this.container) { element.initKalmanFilter(); } };
        this.setUseKalmanFilter = (useKalmanFilter) => { for (let element of this.container) { element.setUseKalmanFilter(useKalmanFilter); } };
        this.add = (poseElement) => { this.container.push(poseElement); };
        this.configure = (poseElementConfigures) => {
            this.container = [];
            poseElementConfigures.forEach(config => {
                if (!config) return;
                let type = config.type;
                if (type === 'Landmark') this.add(new Landmark(this, config.poseElementConfig));
                else if (type === 'VirtualLandmark') this.add(new VirtualLandmark(this, config.poseElementConfig, config.virtualLandmarkConfig));
                else if (type === 'VirtualBone') this.add(new VirtualBone(this, config.poseElementConfig, config.virtualBoneConfig, config.otherName));
                else if (type === 'Bone') this.add(new Bone(this, config.poseElementConfig, config.otherName));
            });
        };
        this.update = (data = null) => { this.container.forEach(poseElement => { poseElement.update(data); }); };
        this.setFromArray = (dataArray, format) => {
            if (!dataArray) return;
            for (let poseEl of this.container) {
                let index = poseEl.index;
                let data = dataArray[index];
                if (data) {
                    if (format === 'xyzv') poseEl.setFromXYZV(data);
                    else if (format === 'array') poseEl.setFromArray(data);
                    else if (format === 'mat') poseEl.setFromMat(data);
                } else { poseEl.update(); }
            }
        };
        this.get = (index) => {
            for (let i = 0; i < this.container.length; i++) {
                if (this.container[i].index === index) return this.container[i];
            }
            return null;
        };
        this.getByName = (name) => {
            for (let poseElement of this.container) {
                if (poseElement.name === name) return poseElement;
            }
            return null;
        };
    }
}

class BoneManager extends PoseElementManager {
    constructor() {
        super();
        this.avatar = null;
        this.avatarType = null;
        this.avatarBones = []; 
        let slerpRatio = 0.3;

        this.setSlerpRatio = (ratio) => { slerpRatio = parseFloat(ratio); }

        this.bindAvatar = (avatar, type, coordinateSystem = null) => {
            this.avatar = avatar.scene || avatar; 
            this.avatarType = type || 'RPM';
            this.avatarBones = [];
            this.coordinateSystem = coordinateSystem;

            console.log(`[BoneManager] 🔥 바인딩 모드: ${this.avatarType}`);

            // 1. 모델 내 모든 뼈 수집
            const bonesInModel = [];
            this.avatar.traverse(obj => {
                if (obj.isBone || obj.type === 'Bone') bonesInModel.push(obj);
            });
            console.log(`[BoneManager] 모델 내 실제 뼈 발견: ${bonesInModel.length}개`);

            // 2. 바인딩 로직 강화
            this.container.forEach(bone => {
                if (!bone) return;
                
                // MediapipeConfig에서 정의한 기본 이름 (Hips, Spine 등)
                const baseKey = (bone.poseElementConfig && bone.poseElementConfig.name) ? bone.poseElementConfig.name.toLowerCase() : bone.name.toLowerCase();
                
                let targetBone = null;

                // Mixamo일 때: 이름에 키워드가 포함되어 있는지 확인 (mixamorig6Hips 등 대응)
                if (this.avatarType === 'Mixamo') {
                    targetBone = bonesInModel.find(b => b.name.toLowerCase().includes(baseKey));
                } else {
                    // RPM 등은 기존 방식
                    const rpmName = (bone.otherName && bone.otherName.RPM) ? bone.otherName.RPM : baseKey;
                    targetBone = this.avatar.getObjectByName(rpmName);
                }

                if (targetBone) {
                    this.avatarBones[bone.index] = targetBone;
                    targetBone.rotation.order = 'YXZ';
                    console.log(`[SUCCESS] ${baseKey} -> ${targetBone.name} 매칭됨`);
                }
            });

            console.log(`[BoneManager] 최종 바인딩 성공: ${Object.keys(this.avatarBones).length}개`);
        };

        this.updateVisibility = () => {
            this.container.forEach(bone => {
                if (bone && typeof bone.updateVisibility === 'function') bone.updateVisibility();
            });
        };

        this.updateAvatar = () => {
            if (!this.avatar || !this.avatarBones || Object.keys(this.avatarBones).length === 0) return;
            
            let avatarWorldQuaternion = new THREE.Quaternion();
            let coordinateSystemInverseQuaternion = new THREE.Quaternion();

            if (this.coordinateSystem) {
                this.coordinateSystem.getWorldQuaternion(coordinateSystemInverseQuaternion);
                coordinateSystemInverseQuaternion.invert();
            }

            this.container.forEach((bone, i) => {
                let avatarBone = this.avatarBones[i];
                if (avatarBone && avatarBone.parent) {
                    avatarBone.parent.getWorldQuaternion(avatarWorldQuaternion);
                    if (this.coordinateSystem) {
                        avatarWorldQuaternion.premultiply(coordinateSystemInverseQuaternion);
                    }
                    let avatarLocalQuaternion = avatarWorldQuaternion.invert().multiply(bone.worldQuaternion);
                    const vis = (bone.visibility !== undefined) ? bone.visibility : 1.0;
                    avatarBone.quaternion.slerp(avatarLocalQuaternion, slerpRatio * vis);
                }
            });
        };

        this.getAvatarRoot = () => { return this.avatar; };
        this.getWorldQuaternionArray = () => {
            let worldQuaternionArray = [];
            for (let bone of this.container) worldQuaternionArray.push(bone.worldQuaternion.toArray());
            return worldQuaternionArray;
        };
    }
}

class VirtualBoneManager extends BoneManager {
    constructor(landmarkManager) {
        super();
        this.landmarkManager = landmarkManager;
    }
}
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

modelPointsPath = '../target_point_clouds/Optical/reg-Points'
load_landmarks_path = '../target_point_clouds/Optical'
numberOfPoints = -1

systemOP = 'op'
systemEM = 'em'
modeAN = 'anatomical'
modeRR = 'regRings'

userNumber = 1
skullNumber = 3
system = systemOP
mode = modeRR
numberOfSutures = 5
userNumberOfRR = 51

useSavedMatrix = True

def createModelPoints(skullNumber):
    '''
    load model source points of the target skull
    '''

    # load points
    modelRegPoints = np.loadtxt('{}\\sk{}_{}points.txt'.format(modelPointsPath, skullNumber, numberOfPoints))

    # flip axes to match acquired points in unity
    for i, point in enumerate(modelRegPoints):
        modelRegPoints[i, :] = [- modelRegPoints[i, 0], modelRegPoints[i, 1], modelRegPoints[i, 2]]

    return modelRegPoints

def acquireRealPoints(skullNumber):

    '''
        load the target points acquired using the optical pointer for the target skull
    '''

    realRegPoints = []

    # load optical acquired points and transform them to the reference marker coordinate system
    for i in range(1, numberOfPoints + 1):
        pose1_fileName = '{}\\p{}.csv'.format(load_landmarks_path, i)
        p_df = pd.read_csv(pose1_fileName)

        all_points = []
        for j in p_df.index:
            p_pos = np.array([[-p_df['Ty'][j]], [-p_df['Tz'][j]], [-p_df['Tx'][j]]])
            p_rot = np.array([p_df['Qy'][j], p_df['Qz'][j], p_df['Qx'][j], p_df['Q0'][j]])

            txp = Rotation.from_quat(p_rot)
            txp = txp.as_matrix()
            txp = np.concatenate([txp, p_pos], axis=1)
            txp = np.concatenate([txp, [[0, 0, 0, 1]]], axis=0)

            r_pos = np.array([[-p_df['Ty.5'][j]], [-p_df['Tz.5'][j]], [-p_df['Tx.5'][j]]])

            r_rot = np.array([p_df['Qy.1'][j], p_df['Qz.1'][j], p_df['Qx.1'][j], p_df['Q0.1'][j]])

            txr = Rotation.from_quat(r_rot)
            txr = txr.as_matrix()
            txr = np.concatenate([txr, r_pos], axis=1)
            txr = np.concatenate([txr, [[0, 0, 0, 1]]], axis=0)

            txRp = np.matmul(np.linalg.inv(txr), txp)

            point = txRp[:-1, -1]
            all_points.append(point)

        realRegPoints.append(np.mean(all_points, axis=0))

    return realRegPoints

def performRegistration(modelPoints, realPoints):

    '''
        Perform landmark registration using unit quaternion
    '''

    # check point clouds sizes
    if len(modelPoints) != len(realPoints):
        raise Exception(' number of points not equal')

    # center point clouds
    size = len(modelPoints)
    ref_c = [0, 0, 0]
    pos_c = [0, 0, 0]

    for i, point in enumerate(modelPoints):
        ref_c = ref_c + modelPoints[i]
        pos_c = pos_c + realPoints[i]

    ref_c = ref_c / size
    pos_c = pos_c / size

    m = np.zeros([3, 3])

    for i, point in enumerate(modelPoints):

        ref_p = modelPoints[i] - ref_c
        pos_p = realPoints[i] - pos_c

        # find the covariance and symmetric matrix
        for k in range(0, 3):

            for l in range(0, 3):
                m[k, l] = m[k, l] + (ref_p[k] * pos_p[l])

    n = np.zeros([4, 4])
    n[0, 0] = +m[0, 0] + m[1, 1] + m[2, 2]
    n[1, 1] = +m[0, 0] - m[1, 1] - m[2, 2]
    n[2, 2] = -m[0, 0] + m[1, 1] - m[2, 2]
    n[3, 3] = -m[0, 0] - m[1, 1] + m[2, 2]
    n[0, 1] = m[1, 2] - m[2, 1]
    n[1, 0] = n[0, 1]
    n[0, 2] = m[2, 0] - m[0, 2]
    n[2, 0] = n[0, 2]
    n[0, 3] = m[0, 1] - m[1, 0]
    n[3, 0] = n[0, 3]
    n[1, 2] = m[0, 1] + m[1, 0]
    n[2, 1] = n[1, 2]
    n[1, 3] = m[2, 0] + m[0, 2]
    n[3, 1] = n[1, 3]
    n[2, 3] = m[1, 2] + m[2, 1]
    n[3, 2] = n[2, 3]

    # SVD decomposition to find the eigen vector with optimal quaternion
    [u, s, vh] = np.linalg.svd(n, hermitian=True)
    quat = np.array([-1]) * [u[1, 0], u[2, 0], u[3, 0], u[0, 0]]

    # get rotation matrix from quaternions
    marker_rotTx = Rotation.from_quat(quat)
    tx_registration = marker_rotTx.as_matrix()

    # Find translation part of transformation
    commTrans = np.zeros(3)
    for i, point in enumerate(modelPoints):
        transformedPoint = realPoints[i] - np.matmul(tx_registration, modelPoints[i])
        commTrans = commTrans + transformedPoint

    centerPoint = 1/len(modelPoints) * commTrans
    pos = [[centerPoint[0]], [centerPoint[1]], [centerPoint[2]]]

    # construct transformation matrix
    tx_registration = np.concatenate([tx_registration, pos], axis=1)
    tx_registration = np.concatenate([tx_registration, [[0, 0, 0, 1]]], axis=0)

    return tx_registration

def plot_points(points_array, title):
    '''
        plot landmark registration
    '''

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(points_array[0, 0], points_array[0, 1], points_array[0, 2], color='red')
    ax.scatter(points_array[1, 0], points_array[1, 1], points_array[1, 2], color='blue')
    ax.scatter(points_array[2, 0], points_array[2, 1], points_array[2, 2], color='green')
    ax.scatter(points_array[3, 0], points_array[3, 1], points_array[3, 2], color='purple')
    ax.scatter(points_array[4, 0], points_array[4, 1], points_array[4, 2], color='black')
    ax.scatter(points_array[5, 0], points_array[5, 1], points_array[5, 2], color='orange')
    if not mode == 'em_anatomical' and numberOfPoints > 6:
        if skullNumber > 0:
            ax.scatter(points_array[6, 0], points_array[6, 1], points_array[6, 2], color='orange')
            ax.scatter(points_array[7, 0], points_array[7, 1], points_array[7, 2], color='orange')
            if not skullNumber == 3:
                ax.scatter(points_array[8, 0], points_array[8, 1], points_array[8, 2], color='orange')
                # ax.scatter(rp[9, 0], rp[9, 1], rp[9, 2], color='orange')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(title)

    plt.show()

def calculateFRE(modelPoints, realPoints, tx_registration):
    '''
        Calculate Fiducial registration error for the landmarks
    '''
    all_FRE = []
    transPoints = []

    # transform source points and calculate distance error with target points
    for i, mp in enumerate(modelPoints):
        mp = np.concatenate([modelPoints[i], [1]], axis=0)
        trans_mp = np.matmul(tx_registration, mp)

        dist_mp = abs(trans_mp[:-1] - realPoints[i])
        transPoints.append(trans_mp)
        all_FRE.append(dist_mp)

    avg_dist = np.mean(all_FRE, axis=0)
    std_dist = np.std(all_FRE, axis=0)

    mp = np.asarray(modelPoints)
    rp = np.asarray(realPoints)
    tp = np.asarray(transPoints)

    plot_points(mp, 'model points')
    plot_points(rp, 'real points')
    plot_points(tp, 'translated points')

    return avg_dist, std_dist

def DigitizeSutuerPoints(tx_RigidCalib, sutureNumber):

    '''
         Digitize sutures/surface points in the preoperative model coordinate system
    '''

    suturePoints = []
    targetCloudPoints = []

    # load suture/surface points
    pose1_fileName = '{}\\pSutureLine{}.csv'.format(load_landmarks_path, sutureNumber)
    p_df = pd.read_csv(pose1_fileName)
    for j in p_df.index:
        try:
            p_pos = np.array([[-p_df['Ty'][j]], [-p_df['Tz'][j]], [-p_df['Tx'][j]]])
            p_rot = np.array([p_df['Qy'][j], p_df['Qz'][j], p_df['Qx'][j], p_df['Q0'][j]])

            # suture/surface point transformation
            txp = Rotation.from_quat(p_rot)
            txp = txp.as_matrix()
            txp = np.concatenate([txp, p_pos], axis=1)
            txp = np.concatenate([txp, [[0, 0, 0, 1]]], axis=0)

            if system == 'em':
                r_pos = np.array([[-p_df['Ty.1'][j]], [-p_df['Tz.1'][j]], [-p_df['Tx.1'][j]]])
            else:  # system == 'op'
                r_pos = np.array([[-p_df['Ty.5'][j]], [-p_df['Tz.5'][j]], [-p_df['Tx.5'][j]]])

            r_rot = np.array([p_df['Qy.1'][j], p_df['Qz.1'][j], p_df['Qx.1'][j], p_df['Q0.1'][j]])

            # reference marker transformation
            txr = Rotation.from_quat(r_rot)
            txr = txr.as_matrix()
            txr = np.concatenate([txr, r_pos], axis=1)
            txr = np.concatenate([txr, [[0, 0, 0, 1]]], axis=0)

            # get suture/surface point in reference marker space
            txRp = np.matmul(np.linalg.inv(txr), txp)
            # get suture/surface point in CT model space using registration matrix
            txMp = np.matmul(np.linalg.inv(tx_RigidCalib), txRp)

            # suture/surface points before registration
            point = txMp[:-1, -1]
            point = [-point[0], point[1], point[2]]
            suturePoints.append(point)

            # suture/surface points after registration
            targetPoint = txRp[:-1, -1]
            targetPoint = [-targetPoint[0], targetPoint[1], targetPoint[2]]
            targetCloudPoints.append(targetPoint)

        except:
            print('{}, sutureLine = {}'.format(j, sutureNumber))
            continue

    np.savetxt('{}\\reg_pc{}.txt'.format(load_landmarks_path, sutureNumber), suturePoints)
    np.savetxt('{}\\pc{}.txt'.format(load_landmarks_path, sutureNumber), targetCloudPoints)

    return 0

if __name__ == '__main__':

    users = [1]
    skulls = [2]
    systems = [systemOP]
    modes = [modeRR]

    for userNumber in users:
        print(f'user = {userNumber}')

        for skullNumber in skulls:
            load_landmarks_path = '{}\\points{}\\sk{}'.format(load_landmarks_path, userNumber, skullNumber)

            if mode == 'regRings':
                if skullNumber == 1 or skullNumber == 2:
                    numberOfPoints = 10
                else:  # skullNumber ==3
                    numberOfPoints = 8

            # get source points
            modelPoints = createModelPoints(skullNumber)
            # get target points
            realPoints = acquireRealPoints(skullNumber)
            # get registration matrix
            tx_registration = performRegistration(modelPoints, realPoints)
            # calculate registration error
            avgError, stdError = calculateFRE(modelPoints, realPoints, tx_registration)
            rmse = np.sqrt(avgError[0] * avgError[0] + avgError[1] * avgError[1] + avgError[2] * avgError[2])

            print("registration rms_mean = ", avgError)
            print("registration rms_std = ", stdError)
            print("###########################################")
            print("saving calibration matrix to file: {}".format('{}\\regMatrix.csv'.format(load_landmarks_path)))
            regTx_df = {'c1': tx_registration[:, 0], 'c2': tx_registration[:, 1], 'c3': tx_registration[:, 2], 'c4': tx_registration[:, 3]}
            ctx = pd.DataFrame(regTx_df, columns=['c1', 'c2', 'c3', 'c4'])
            ctx.to_csv('{}\\regMatrix.csv'.format(load_landmarks_path))

            print("saving rmse to file: {}".format(rmse))
            rmse_df = {'registrationRMSE': [rmse]}
            rmseDF = pd.DataFrame(rmse_df, columns=['registrationRMSE'])
            rmseDF.to_csv('{}\\rmse.csv'.format(load_landmarks_path))

            # get surface/suture points in source model coordinate system
            for i in range(1, numberOfSutures + 1):
                suturePoints = DigitizeSutuerPoints(tx_registration, i)

    print("finished processing...")
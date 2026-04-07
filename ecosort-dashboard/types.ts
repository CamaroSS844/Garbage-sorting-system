
export enum DeviceStatus {
  ONLINE = 'online',
  OFFLINE = 'offline'
}

export interface Device {
  id: string;
  name: string;
  status: DeviceStatus;
  lastSeen: string;
  location: string;
}

export interface FailedInference {
  id: string;
  timestamp: string;
  imageUrl: string;
  confidence: number;
  originalGuess: string;
  deviceId: string;
  reviewed: boolean;
}

export type TrashCategory = 'Plastic' | 'Paper' | 'Metal' | 'Glass' | 'Organic' | 'Rejected';

export interface ReportStats {
  totalProcessed: number;
  accepted: number;
  rejected: number;
  failedInference: number;
  accuracy: number;
  uptime: number;
}

export interface InferenceDetection {
  bbox: [number, number, number, number];
  confidence: number;
  class_id?: number;
  class_name?: string;
  class?: string; // Cloud mode uses 'class'
  track_id?: number;
}

export interface InferenceMessage {
  type: 'detection' | 'tracked' | 'camera_status';
  timestamp?: number;
  mode?: 'local' | 'cloud';
  objects?: InferenceDetection[];
  detections?: InferenceDetection[]; // Cloud mode uses 'detections'
  inference_time_ms?: number;
  connected?: boolean;
}

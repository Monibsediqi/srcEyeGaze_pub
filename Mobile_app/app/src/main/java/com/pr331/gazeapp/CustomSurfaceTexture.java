package com.pr331.gazeapp;

import android.graphics.SurfaceTexture;

public class CustomSurfaceTexture extends SurfaceTexture {
    public CustomSurfaceTexture(int texName) {
        super(texName);
        init();
    }

    private void init() {
        super.detachFromGLContext();
    }
}

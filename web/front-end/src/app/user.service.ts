import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class UserService {
  private _username: string | null = null;

  get username(): string | null {
    if (!this._username) {
      this._username = localStorage.getItem('username');
    }
    return this._username;
  }

  set username(value: string | null) {
    this._username = value;
    if (value) {
      localStorage.setItem('username', value);
    } else {
      localStorage.removeItem('username');
    }
  }
}

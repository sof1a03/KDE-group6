import { TestBed } from '@angular/core/testing';

import { UserlikeService } from './userlike.service';

describe('UserlikeService', () => {
  let service: UserlikeService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(UserlikeService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
